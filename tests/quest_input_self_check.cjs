'use strict';

const assert = require('node:assert/strict');
const questInput = require('../yam_control/teleop/web/quest_input.js');

/**
 * Fake WebXR pose와 Gamepad가 기존 xr_frame payload로 유지되는지 검증한다.
 *
 * @returns {void}
 */
function main(
) {
  const gripSpace = {};
  const referenceSpace = {};
  const controllerPose = {
    transform: {
      position: {x: 1, y: 2, z: 3},
      orientation: {x: 0, y: 0, z: 0, w: 1},
    },
  };
  const viewerPose = {
    transform: {
      position: {x: 4, y: 5, z: 6},
      orientation: {x: 0, y: 1, z: 0, w: 0},
    },
  };
  const frame = {
    getPose(
      actualGripSpace,
      actualReferenceSpace,
    ) {
      assert.equal(actualGripSpace, gripSpace);
      assert.equal(actualReferenceSpace, referenceSpace);
      return controllerPose;
    },
    getViewerPose(
      actualReferenceSpace,
    ) {
      assert.equal(actualReferenceSpace, referenceSpace);
      return viewerPose;
    },
  };
  const inputSource = {
    handedness: 'right',
    gripSpace,
    gamepad: {
      buttons: [
        {pressed: true, touched: true, value: 0.75},
        {pressed: false, touched: false, value: 0},
      ],
      axes: [0.25, -0.5],
    },
  };

  const readout = questInput.readController(
    inputSource,
    frame,
    referenceSpace,
    [0, 0, 0.05],
  );
  assert.deepEqual(readout, {
    handedness: 'right',
    rawPosition: [1, 2, 3],
    orientation: [0, 0, 0, 1],
    controller: {
      position: [1, 2, 3.05],
      orientation: [0, 0, 0, 1],
      buttons: [
        {p: true, t: true, v: 0.75},
        {p: false, t: false, v: 0},
      ],
      axes: [0.25, -0.5],
    },
  });
  assert.deepEqual(
    questInput.readViewer(frame, referenceSpace),
    {
      position: [4, 5, 6],
      orientation: [0, 1, 0, 0],
    },
  );
  assert.equal(
    questInput.readController(
      {...inputSource, handedness: 'none'},
      frame,
      referenceSpace,
      [0, 0, 0],
    ),
    null,
  );
}

main();
