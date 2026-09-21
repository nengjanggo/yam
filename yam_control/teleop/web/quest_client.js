(() => {
  'use strict';

  const status = document.getElementById('status');
  const button = document.getElementById('teleop-button');
  const video = document.getElementById('camera-preview');
  const canvas = document.getElementById('xr-canvas');
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
        gl_FragColor = vec4(color.rgb, mix(1.0, chromaAlpha, uUseChromaKey));
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
    videoTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
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
    const panelAnchor = cameraLayout === 'wrist'
      ? anchorFromViewer(viewerPose)
      : videoAnchor;
    if (!panelAnchor) {
      return;
    }
    gl.useProgram(videoProgram);
    gl.bindTexture(gl.TEXTURE_2D, videoTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.disable(gl.DEPTH_TEST);
    // MuJoCo는 큰 world-locked panel, wrist camera는 작은 head-locked 오른쪽 아래 panel을 사용
    const panelTransform = cameraLayout === 'wrist'
      ? new Float32Array([
        0.28, 0, 0, 0,
        0, 0.21, 0, 0,
        0, 0, 1, 0,
        0.30, -0.22, -0.70, 1,
      ])
      : new Float32Array([
        0.65, 0, 0, 0,
        0, 0.49, 0, 0,
        0, 0, 1, 0,
        0, -0.75, -1.0, 1,
      ]);
    const model = multiplyMatrices(panelAnchor, panelTransform);
    gl.uniform1f(
      gl.getUniformLocation(videoProgram, 'uUseChromaKey'),
      cameraLayout === 'mujoco' ? 1.0 : 0.0,
    );
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
    if (message.type === 'webrtc_offer') {
      acceptOffer(message).catch(() => {
        status.textContent = 'Camera preview 연결에 실패했습니다.';
      });
    }
  });
  button.addEventListener('click', toggleTeleop);
  selectXrMode();
})();
