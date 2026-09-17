(function initialiseQuestInput(globalScope) {
  'use strict';

  /**
   * WebXR DOMPointReadOnly shape `(3,)`을 number array로 변환한다.
   *
   * @param {DOMPointReadOnly} vector
   * @returns {[number, number, number]}
   */
  function vectorToArray(
    vector,
  ) {
    return [vector.x, vector.y, vector.z];
  }

  /**
   * WebXR DOMPointReadOnly quaternion shape `(4,)`을 xyzw number array로 변환한다.
   *
   * @param {DOMPointReadOnly} quaternion
   * @returns {[number, number, number, number]}
   */
  function quaternionToArray(
    quaternion,
  ) {
    return [quaternion.x, quaternion.y, quaternion.z, quaternion.w];
  }

  /**
   * Controller local offset shape `(3,)`을 xyzw quaternion으로 회전해 world offset shape `(3,)`으로 반환한다.
   *
   * @param {[number, number, number]} vector
   * @param {[number, number, number, number]} quaternion
   * @returns {[number, number, number]}
   */
  function rotateVectorByQuaternion(
    vector,
    quaternion,
  ) {
    const [vectorX, vectorY, vectorZ] = vector;
    const [quaternionX, quaternionY, quaternionZ, quaternionW] = quaternion;
    const crossX = quaternionY * vectorZ - quaternionZ * vectorY;
    const crossY = quaternionZ * vectorX - quaternionX * vectorZ;
    const crossZ = quaternionX * vectorY - quaternionY * vectorX;
    const secondCrossX = quaternionY * crossZ - quaternionZ * crossY;
    const secondCrossY = quaternionZ * crossX - quaternionX * crossZ;
    const secondCrossZ = quaternionX * crossY - quaternionY * crossX;
    return [
      vectorX + 2 * quaternionW * crossX + 2 * secondCrossX,
      vectorY + 2 * quaternionW * crossY + 2 * secondCrossY,
      vectorZ + 2 * quaternionW * crossZ + 2 * secondCrossZ,
    ];
  }

  /**
   * XRInputSource gripSpace와 Gamepad를 기존 xr_frame controller payload로 변환한다.
   *
   * @param {XRInputSource} inputSource
   * @param {XRFrame} frame
   * @param {XRReferenceSpace} referenceSpace
   * @param {[number, number, number]} positionOffset
   * @returns {object | null}
   */
  function readController(
    inputSource,
    frame,
    referenceSpace,
    positionOffset,
  ) {
    if (!['left', 'right'].includes(inputSource.handedness) || !inputSource.gripSpace) {
      return null;
    }
    const pose = frame.getPose(inputSource.gripSpace, referenceSpace);
    if (!pose) {
      return null;
    }

    // Shape `(3,)` position과 shape `(4,)` xyzw quaternion을 WebXR pose에서 추출
    const rawPosition = vectorToArray(pose.transform.position);
    const orientation = quaternionToArray(pose.transform.orientation);
    const worldOffset = rotateVectorByQuaternion(positionOffset, orientation);
    const position = [
      rawPosition[0] + worldOffset[0],
      rawPosition[1] + worldOffset[1],
      rawPosition[2] + worldOffset[2],
    ];
    const gamepad = inputSource.gamepad;
    const buttons = gamepad
      ? Array.from(gamepad.buttons, (button) => ({
        p: button.pressed,
        t: button.touched,
        v: button.value,
      }))
      : [];
    const axes = gamepad ? Array.from(gamepad.axes) : [];
    return {
      handedness: inputSource.handedness,
      rawPosition,
      orientation,
      controller: {
        position,
        orientation,
        buttons,
        axes,
      },
    };
  }

  /**
   * XRFrame viewer pose를 기존 xr_frame viewer payload로 변환한다.
   *
   * @param {XRFrame} frame
   * @param {XRReferenceSpace} referenceSpace
   * @returns {object | null}
   */
  function readViewer(
    frame,
    referenceSpace,
  ) {
    const pose = frame.getViewerPose(referenceSpace);
    if (!pose) {
      return null;
    }
    return {
      position: vectorToArray(pose.transform.position),
      orientation: quaternionToArray(pose.transform.orientation),
    };
  }

  const questInput = Object.freeze({
    readController,
    readViewer,
  });
  globalScope.YamQuestInput = questInput;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = questInput;
  }
}(globalThis));
