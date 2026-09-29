(() => {
  'use strict';

  const status = document.getElementById('status');
  const button = document.getElementById('teleop-button');
  const video = document.getElementById('camera-preview');
  const canvas = document.getElementById('xr-canvas');
  const hudCanvas = document.getElementById('hud-canvas');
  const hudContext = hudCanvas.getContext('2d');
  const websocketUrl = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`;
  const websocket = new WebSocket(websocketUrl);
  const xrMode = 'immersive-ar';
  let xrSession = null;
  let xrReferenceSpace = null;
  let gl = null;
  let peerConnection = null;
  let videoProgram = null;
  let videoTexture = null;
  let videoBuffer = null;
  let videoAnchor = null;
  let cameraLayout = 'none';
  let lastFrameSentAt = -Infinity;
  let hudTexture = null;
  let hudTextureKey = '';
  let episodeStatus = null;
  let episodeStatusReceivedAt = -Infinity;

  // Python process가 멈추면 오래된 알림이 남지 않도록 일정 시간 갱신이 없으면 숨김
  const HUD_STALE_MS = 10000;
  const HUD_CANVAS_WIDTH = 1280;
  const HUD_CANVAS_HEIGHT = 320;
  const HUD_LINE_HEIGHT = 88;
  const HUD_PADDING = 16;
  const HUD_COLORS = {
    text: '#f3f6f8',
    muted: '#b8c2cc',
    success: '#74d4bd',
    warning: '#ffd166',
    failure: '#ff7b72',
  };
  const RESULT_TEXT = {
    succeeded: '에피소드 성공',
    finished: '에피소드 종료',
    discarded: '에피소드 실패',
    aborted: '에피소드 중단',
  };
  hudCanvas.width = HUD_CANVAS_WIDTH;
  hudCanvas.height = HUD_CANVAS_HEIGHT;

  /** WebSocket이 열려 있을 때만 JSON message를 보낸다. */
  function send(
    message,
  ) {
    if (websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify(message));
    }
  }

  /** Quest Browser가 지원하는 immersive WebXR mode를 선택한다. */
  async function selectXrMode() {
    if (!navigator.xr) {
      status.textContent = 'WebXR을 사용할 수 없습니다.';
      return;
    }
    if (!await navigator.xr.isSessionSupported(xrMode)) {
      status.textContent = 'Passthrough WebXR을 사용할 수 없습니다.';
      return;
    }
    button.disabled = websocket.readyState !== WebSocket.OPEN;
  }

  /** Relay offer에 응답하고 선택된 video track을 preview에 연결한다. */
  async function acceptOffer(
    offer,
  ) {
    if (peerConnection) {
      peerConnection.close();
    }
    peerConnection = new RTCPeerConnection({ iceServers: [] });
    peerConnection.addEventListener('track', (event) => {
      if (event.streams[0]) {
        video.srcObject = event.streams[0];
        video.play().catch(() => {
          status.textContent = 'Camera preview 자동 재생에 실패했습니다.';
        });
      }
    });
    await peerConnection.setRemoteDescription({ type: offer.sdp_type, sdp: offer.sdp });
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);
    send({
      type: 'webrtc_answer',
      sdp: peerConnection.localDescription.sdp,
      sdp_type: peerConnection.localDescription.type,
    });
  }

  /** 현재 WebRTC connection과 video frame을 비우고 panel을 숨긴다. */
  function clearVideo() {
    if (peerConnection) {
      peerConnection.close();
      peerConnection = null;
    }
    video.srcObject = null;
    videoAnchor = null;
    cameraLayout = 'none';
  }

  /** Shape `(4, 4)` column-major matrix 두 개를 곱한다. */
  function multiplyMatrices(
    left,
    right,
  ) {
    const result = new Float32Array(16);
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        for (let index = 0; index < 4; index += 1) {
          result[row + 4 * column] += left[row + 4 * index] * right[index + 4 * column];
        }
      }
    }
    return result;
  }

  /** 첫 HMD pose에서 yaw만 사용해 world-locked panel 기준 행렬을 만든다. */
  function anchorFromViewer(
    viewerPose,
  ) {
    const position = viewerPose.transform.position;
    const orientation = viewerPose.transform.orientation;
    const forwardX = -2 * (orientation.x * orientation.z + orientation.y * orientation.w);
    const forwardZ = -(1 - 2 * (orientation.x ** 2 + orientation.y ** 2));
    const length = Math.hypot(forwardX, forwardZ);
    const x = length < 1e-4 ? 0 : forwardX / length;
    const z = length < 1e-4 ? -1 : forwardZ / length;
    return new Float32Array([
      -z, 0, x, 0,
      0, 1, 0, 0,
      -x, 0, -z, 0,
      position.x, position.y, position.z, 1,
    ]);
  }

  /** WebGL shader를 생성하고 compile 결과를 확인한다. */
  function compileShader(
    type,
    source,
  ) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  /** Video를 passthrough 위에 합성할 WebGL resource를 준비한다. */
  function prepareVideo() {
    const vertexShader = compileShader(gl.VERTEX_SHADER, `
      attribute vec4 aVertex;
      uniform mat4 uMvp;
      varying vec2 vUv;
      void main() {
        gl_Position = uMvp * vec4(aVertex.xy, 0.0, 1.0);
        vUv = aVertex.zw;
      }
    `);
    const fragmentShader = compileShader(gl.FRAGMENT_SHADER, `
      precision mediump float;
      uniform sampler2D uVideo;
      uniform float uUseChromaKey;
      varying vec2 vUv;
      void main() {
        vec4 color = texture2D(uVideo, vUv);
        float brightness = max(max(color.r, color.g), color.b);
        float chromaAlpha = smoothstep(0.015, 0.06, brightness);
        gl_FragColor = vec4(color.rgb, color.a * mix(1.0, chromaAlpha, uUseChromaKey));
      }
    `);
    videoProgram = gl.createProgram();
    gl.attachShader(videoProgram, vertexShader);
    gl.attachShader(videoProgram, fragmentShader);
    gl.linkProgram(videoProgram);
    if (!gl.getProgramParameter(videoProgram, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(videoProgram));
    }
    // Shape `(6, 4)`는 두 triangle의 x, y, u, v를 포함한다.
    videoBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, videoBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -0.5, -0.5, 0, 0, 0.5, -0.5, 1, 0, 0.5, 0.5, 1, 1,
      -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, -0.5, 0.5, 0, 1,
    ]), gl.STATIC_DRAW);
    videoTexture = createPanelTexture();
    hudTexture = createPanelTexture();
    hudTextureKey = '';
  }

  /** Panel image를 담을 linear filtering texture를 생성한다. */
  function createPanelTexture() {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  /** Texture를 upload하고 model 행렬 위치의 quad를 모든 eye view에 그린다. */
  function drawPanel(
    viewerPose,
    layer,
    texture,
    source,
    model,
    useChromaKey,
  ) {
    gl.useProgram(videoProgram);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    if (source) {
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
    }
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    gl.uniform1f(gl.getUniformLocation(videoProgram, 'uUseChromaKey'), useChromaKey ? 1.0 : 0.0);
    gl.bindBuffer(gl.ARRAY_BUFFER, videoBuffer);
    const vertexLocation = gl.getAttribLocation(videoProgram, 'aVertex');
    gl.enableVertexAttribArray(vertexLocation);
    gl.vertexAttribPointer(vertexLocation, 4, gl.FLOAT, false, 0, 0);
    for (const view of viewerPose.views) {
      const viewport = layer.getViewport(view);
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      const viewProjection = multiplyMatrices(view.projectionMatrix, view.transform.inverse.matrix);
      const mvp = multiplyMatrices(viewProjection, model);
      gl.uniformMatrix4fv(gl.getUniformLocation(videoProgram, 'uMvp'), false, mvp);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }

  /** Video frame을 target별 panel layout으로 passthrough 위에 그린다. */
  function drawVideo(
    viewerPose,
    layer,
  ) {
    if (cameraLayout === 'none' || video.readyState < 2 || !video.videoWidth) {
      return;
    }
    if (cameraLayout === 'mujoco' && !videoAnchor) {
      videoAnchor = anchorFromViewer(viewerPose);
    }
    // Wrist panel은 고개를 숙여도 보이도록 yaw뿐 아니라 pitch까지 머리 방향을 그대로 따라감
    const panelAnchor = cameraLayout === 'wrist'
      ? viewerPose.transform.matrix
      : videoAnchor;
    if (!panelAnchor) {
      return;
    }
    // MuJoCo는 큰 world-locked panel, wrist camera는 작은 head-locked 시야 왼쪽 아래 panel을 사용
    const panelTransform = cameraLayout === 'wrist'
      ? new Float32Array([
        0.42, 0, 0, 0,
        0, 0.315, 0, 0,
        0, 0, 1, 0,
        -0.23, -0.1675, -0.70, 1,
      ])
      : new Float32Array([
        0.65, 0, 0, 0,
        0, 0.49, 0, 0,
        0, 0, 1, 0,
        0, -0.75, -1.0, 1,
      ]);
    const model = multiplyMatrices(panelAnchor, panelTransform);
    drawPanel(viewerPose, layer, videoTexture, video, model, cameraLayout === 'mujoco');
  }

  /** 초 단위 시간을 정수 초 문자열로 변환한다. */
  function formatSeconds(
    seconds,
  ) {
    return String(Math.floor(seconds ?? 0));
  }

  /** Episode status를 HUD에 표시할 text, color와 font size 목록으로 변환한다. */
  function episodeStatusLines(
    status,
  ) {
    const lines = [];
    const primary = status.primary_button ?? 'A';
    const secondary = status.secondary_button ?? 'B';
    const limit = formatSeconds(status.limit_s);
    if (status.phase === 'running') {
      lines.push({
        text: `에피소드 진행 중   ${formatSeconds(status.elapsed_s)} / ${limit}초`,
        color: HUD_COLORS.text,
        size: 60,
      });
    } else if (status.phase === 'awaiting_outcome') {
      lines.push({ text: `제한 시간 도달   ${limit} / ${limit}초`, color: HUD_COLORS.warning, size: 60 });
      lines.push({
        text: `${primary}: 성공(저장)    ${secondary}: 실패(저장 안 함)`,
        color: HUD_COLORS.text,
        size: 56,
      });
    } else {
      const resultText = RESULT_TEXT[status.last_result];
      if (resultText) {
        const saved = status.last_episode_saved === true;
        const succeeded = status.last_result === 'succeeded' || status.last_result === 'finished';
        lines.push({
          text: `${resultText}, ${saved ? '저장됨' : '저장되지 않음'}`,
          color: saved ? HUD_COLORS.success : succeeded ? HUD_COLORS.warning : HUD_COLORS.failure,
          size: 60,
        });
      }
      const promptText = {
        saving: '에피소드 저장 중',
        waiting_prepare: `${primary}: 에피소드 초기화`,
        preparing: '에피소드 초기화 중',
        waiting_start: `${primary}: 에피소드 시작`,
        starting: '에피소드 시작 중',
      }[status.phase];
      if (promptText) {
        lines.push({ text: promptText, color: HUD_COLORS.text, size: 56 });
      }
    }
    if (typeof status.saved_episode_count === 'number') {
      lines.push({ text: `저장된 에피소드 ${status.saved_episode_count}개`, color: HUD_COLORS.muted, size: 48 });
    }
    return lines;
  }

  /** HUD 줄 목록을 canvas 위쪽의 반투명 상자에 가운데 정렬로 그린다. */
  function renderHud(
    lines,
  ) {
    hudContext.clearRect(0, 0, HUD_CANVAS_WIDTH, HUD_CANVAS_HEIGHT);
    hudContext.fillStyle = 'rgba(12, 16, 20, 0.75)';
    hudContext.fillRect(0, 0, HUD_CANVAS_WIDTH, lines.length * HUD_LINE_HEIGHT + 2 * HUD_PADDING);
    hudContext.textAlign = 'center';
    hudContext.textBaseline = 'middle';
    lines.forEach((line, index) => {
      hudContext.font = `600 ${line.size}px system-ui, sans-serif`;
      hudContext.fillStyle = line.color;
      hudContext.fillText(
        line.text,
        HUD_CANVAS_WIDTH / 2,
        HUD_PADDING + HUD_LINE_HEIGHT * (index + 0.5),
        HUD_CANVAS_WIDTH - 2 * HUD_PADDING,
      );
    });
  }

  /** 최신 episode status를 시야 중앙 상단에 head-locked 알림으로 그린다. */
  function drawHud(
    viewerPose,
    layer,
  ) {
    if (!episodeStatus || Date.now() - episodeStatusReceivedAt > HUD_STALE_MS) {
      return;
    }
    const lines = episodeStatusLines(episodeStatus);
    if (lines.length === 0) {
      return;
    }
    // 내용이 바뀐 경우에만 canvas를 다시 그리고 texture를 upload
    const key = JSON.stringify(lines);
    const source = key === hudTextureKey ? null : hudCanvas;
    if (source) {
      renderHud(lines);
      hudTextureKey = key;
    }
    // 머리 방향을 그대로 따라가며 시선 약 12도 위, 0.7 m 앞에 표시
    const hudTransform = new Float32Array([
      0.44, 0, 0, 0,
      0, 0.11, 0, 0,
      0, 0, 1, 0,
      0, 0.15, -0.70, 1,
    ]);
    const model = multiplyMatrices(viewerPose.transform.matrix, hudTransform);
    drawPanel(viewerPose, layer, hudTexture, source, model, false);
  }

  /** WebXR frame을 그린 뒤 controller와 HMD pose를 relay로 보낸다. */
  function onXrFrame(
    timestamp,
    frame,
  ) {
    if (!xrSession) {
      return;
    }
    xrSession.requestAnimationFrame(onXrFrame);
    const layer = xrSession.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, xrMode === 'immersive-ar' ? 0 : 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const viewer = YamQuestInput.readViewer(frame, xrReferenceSpace);
    if (!viewer) {
      return;
    }
    const controllers = {};
    for (const inputSource of xrSession.inputSources) {
      const readout = YamQuestInput.readController(
        inputSource,
        frame,
        xrReferenceSpace,
        [0, 0, 0.05],
      );
      if (readout) {
        controllers[readout.handedness] = readout.controller;
      }
    }
    // Controller JSON은 30 Hz로 제한하고 video rendering과 독립적으로 전송한다.
    if (Object.keys(controllers).length > 0 && timestamp - lastFrameSentAt >= 33) {
      send({ type: 'xr_frame', t_client: timestamp, controllers, viewer });
      lastFrameSentAt = timestamp;
    }
    const viewerPose = frame.getViewerPose(xrReferenceSpace);
    if (viewerPose) {
      drawVideo(viewerPose, layer);
      drawHud(viewerPose, layer);
    }
  }

  /** Immersive WebXR session을 시작하거나 종료한다. */
  async function toggleTeleop() {
    if (xrSession) {
      await xrSession.end();
      return;
    }
    gl = canvas.getContext('webgl2', { xrCompatible: true });
    if (!gl) {
      status.textContent = 'WebGL2를 사용할 수 없습니다.';
      return;
    }
    xrSession = await navigator.xr.requestSession(xrMode, { optionalFeatures: ['local-floor'] });
    xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
    xrReferenceSpace = await xrSession.requestReferenceSpace('local-floor').catch(
      () => xrSession.requestReferenceSpace('local'),
    );
    prepareVideo();
    videoAnchor = null;
    xrSession.addEventListener('end', () => {
      xrSession = null;
      xrReferenceSpace = null;
      videoAnchor = null;
      button.textContent = 'Start Teleop';
      status.textContent = 'WebXR session이 종료되었습니다.';
    });
    button.textContent = 'Stop Teleop';
    status.textContent = 'WebXR session 실행 중';
    xrSession.requestAnimationFrame(onXrFrame);
  }

  websocket.addEventListener('open', () => {
    status.textContent = 'Relay 연결됨';
    button.disabled = !xrMode;
  });
  websocket.addEventListener('close', () => {
    status.textContent = 'Relay 연결이 끊겼습니다. 페이지를 새로고침하세요.';
    button.disabled = true;
    if (xrSession) {
      xrSession.end();
    }
  });
  websocket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'camera_list') {
      clearVideo();
      const camera = message.cameras[0];
      if (camera) {
        cameraLayout = camera.layout;
        send({ type: 'webrtc_request', enabled_cameras: [camera.id] });
      }
    }
    if (message.type === 'episode_status') {
      episodeStatus = message.phase === 'idle' ? null : message;
      episodeStatusReceivedAt = Date.now();
    }
    if (message.type === 'webrtc_offer') {
      acceptOffer(message).catch(() => {
        status.textContent = 'Camera preview 연결에 실패했습니다.';
      });
    }
  });
  button.addEventListener('click', toggleTeleop);
  selectXrMode();
})();
