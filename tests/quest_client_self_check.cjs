'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const questInput = require('../yam_control/teleop/web/quest_input.js');

/** Local Quest Web UI가 camera 요청과 xr_frame을 전송하는지 fake browser로 확인한다. */
async function main() {
  const elements = Object.fromEntries(
    ['status', 'teleop-button', 'camera-preview', 'xr-canvas'].map((id) => [id, {
      textContent: '',
      disabled: true,
      readyState: 2,
      videoWidth: 640,
      srcObject: null,
      addEventListener(type, listener) { this[type] = listener; },
      getContext() { return gl; },
    }]),
  );
  let lastMvp = null;
  let lastChromaKey = null;
  const drawnMvps = [];
  const hudTexts = [];
  const hudContext = new Proxy({}, {
    get(target, key) {
      if (key === 'fillText') return (text) => { hudTexts.push(text); };
      return () => undefined;
    },
    set() { return true; },
  });
  elements['hud-canvas'] = { getContext: () => hudContext };
  const gl = new Proxy({}, {
    get(target, key) {
      if (key === 'getShaderParameter' || key === 'getProgramParameter') return () => true;
      if (key === 'uniformMatrix4fv') return (location, transpose, value) => { lastMvp = value; drawnMvps.push(value); };
      if (key === 'uniform1f') return (location, value) => { lastChromaKey = value; };
      if (key.startsWith('create')) return () => ({});
      return () => undefined;
    },
  });
  let websocket = null;
  let xrSession = null;

  class FakeWebSocket {
    static OPEN = 1;

    constructor(url) {
      this.url = url;
      this.readyState = FakeWebSocket.OPEN;
      this.messages = [];
      this.listeners = {};
      websocket = this;
    }

    addEventListener(type, listener) { this.listeners[type] = listener; }

    send(message) { this.messages.push(JSON.parse(message)); }

    emit(type, data = null) { this.listeners[type]({ data: JSON.stringify(data) }); }
  }

  const context = {
    document: { getElementById: (id) => elements[id] },
    location: { protocol: 'https:', host: '192.168.1.2:8443' },
    navigator: { xr: {
      isSessionSupported: async () => true,
      requestSession: async () => {
        xrSession = {
          inputSources: [{
            handedness: 'right',
            gripSpace: {},
            gamepad: { buttons: [{ pressed: false, touched: false, value: 0 }, { pressed: true, touched: true, value: 1 }], axes: [] },
          }],
          updateRenderState(state) { this.renderState = state; },
          requestReferenceSpace: async () => ({}),
          addEventListener() {},
          requestAnimationFrame(callback) { this.onFrame = callback; },
        };
        return xrSession;
      },
    } },
    WebSocket: FakeWebSocket,
    XRWebGLLayer: class {
      constructor() { this.framebuffer = {}; }
      getViewport() { return { x: 0, y: 0, width: 1024, height: 1024 }; }
    },
    RTCPeerConnection: class {},
    Float32Array,
    YamQuestInput: questInput,
  };
  const clientPath = path.join(__dirname, '../yam_control/teleop/web/quest_client.js');
  vm.runInNewContext(fs.readFileSync(clientPath, 'utf8'), context);
  await Promise.resolve();
  websocket.emit('open');
  assert.equal(elements['teleop-button'].disabled, false);
  websocket.emit('message', {
    type: 'camera_list',
    cameras: [{ id: 'wrist', label: 'Wrist Camera', layout: 'wrist' }],
  });
  assert.equal(websocket.messages[0].type, 'webrtc_request');
  assert.deepEqual(Array.from(websocket.messages[0].enabled_cameras), ['wrist']);

  await elements['teleop-button'].click();
  const pose = {
    transform: {
      position: { x: 0, y: 1, z: 0 },
      orientation: { x: 0, y: 0, z: 0, w: 1 },
      matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]),
    },
    views: [{
      projectionMatrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      transform: {
        inverse: {
          matrix: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
        },
      },
    }],
  };
  const frame = {
    getViewerPose: () => pose,
    getPose: () => pose,
  };
  xrSession.onFrame(100, frame);
  assert.equal(websocket.messages[1].type, 'xr_frame');
  assert.equal(websocket.messages[1].controllers.right.buttons[1].p, true);
  assert.deepEqual(Array.from(websocket.messages[1].viewer.position), [0, 1, 0]);
  assert.ok(Math.abs(lastMvp[0] - 0.42) < 1e-6);
  assert.ok(Math.abs(lastMvp[5] - 0.315) < 1e-6);
  assert.ok(lastMvp[12] < 0);
  assert.ok(lastMvp[13] < 1);
  assert.equal(lastChromaKey, 0);
  assert.equal(drawnMvps.length, 1);

  // Episode 결과와 다음 단계 안내를 시야 중앙 상단 알림으로 그린다.
  websocket.emit('message', {
    type: 'episode_status',
    phase: 'waiting_prepare',
    elapsed_s: null,
    limit_s: null,
    last_result: 'succeeded',
    last_episode_saved: true,
    saved_episode_count: 7,
    primary_button: 'A',
    secondary_button: 'B',
  });
  xrSession.onFrame(200, frame);
  assert.deepEqual(hudTexts, ['에피소드 성공, 저장됨', 'A: 에피소드 초기화', '저장된 에피소드 7개']);
  assert.ok(Math.abs(lastMvp[0] - 0.44) < 1e-6);
  assert.ok(Math.abs(lastMvp[12]) < 1e-6);
  assert.ok(lastMvp[13] > 1);
  assert.equal(lastChromaKey, 0);

  // 같은 내용은 canvas를 다시 그리지 않는다.
  xrSession.onFrame(300, frame);
  assert.equal(hudTexts.length, 3);

  hudTexts.length = 0;
  websocket.emit('message', {
    type: 'episode_status',
    phase: 'running',
    elapsed_s: 37.9,
    limit_s: 120,
    last_result: null,
    last_episode_saved: null,
    saved_episode_count: 7,
  });
  xrSession.onFrame(400, frame);
  assert.deepEqual(hudTexts, ['에피소드 진행 중   37 / 120초', '저장된 에피소드 7개']);

  hudTexts.length = 0;
  websocket.emit('message', {
    type: 'episode_status',
    phase: 'awaiting_outcome',
    elapsed_s: 120,
    limit_s: 120,
    saved_episode_count: 7,
    primary_button: 'X',
    secondary_button: 'Y',
  });
  xrSession.onFrame(500, frame);
  assert.deepEqual(hudTexts, [
    '제한 시간 도달   120 / 120초',
    'X: 성공(저장)    Y: 실패(저장 안 함)',
    '저장된 에피소드 7개',
  ]);

  // Idle status를 받으면 알림을 숨긴다.
  drawnMvps.length = 0;
  websocket.emit('message', { type: 'episode_status', phase: 'idle' });
  xrSession.onFrame(600, frame);
  assert.equal(drawnMvps.length, 1);

  websocket.emit('message', { type: 'camera_list', cameras: [] });
  assert.equal(elements['camera-preview'].srcObject, null);
}

main();
