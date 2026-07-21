(function (global) {
  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function safeVec3(x, y, z) {
    return new THREE.Vector3(x || 0, y || 0, z || 0);
  }

  function arrToVec(a) {
    if (!a) return null;
    return new THREE.Vector3(
      (a[0] - 0.5) * 2.0,
      -(a[1] - 0.5) * 2.0,
      -(a[2] || 0) * 2.0
    );
  }

  function getPoseVec(frame, idx) {
    if (!frame || !frame.pose || !frame.pose[String(idx)]) return null;
    return arrToVec(frame.pose[String(idx)]);
  }

  function getHandVec(frame, side, idx) {
    const src = side === "left" ? frame.left_hand : frame.right_hand;
    if (!src || !src[idx]) return null;
    return arrToVec(src[idx]);
  }

  function resolveBone(bones, names) {
    for (const n of names) {
      if (bones[n]) return bones[n];
    }
    return null;
  }

  function slerpBoneToDir(bone, targetDir, amount) {
    if (!bone || !targetDir || targetDir.lengthSq() < 1e-8) return;
    const from = new THREE.Vector3(0, 1, 0);
    const q = new THREE.Quaternion().setFromUnitVectors(from, targetDir.clone().normalize());
    bone.quaternion.slerp(q, amount);
  }

  function solveArm(frame, bones, side) {
    const isLeft = side === "left";

    const shoulderIdx = isLeft ? 11 : 12;
    const elbowIdx = isLeft ? 13 : 14;
    const wristIdx = isLeft ? 15 : 16;

    const shoulder = getPoseVec(frame, shoulderIdx);
    const elbow = getPoseVec(frame, elbowIdx);
    const wrist = getPoseVec(frame, wristIdx);

    if (!shoulder || !elbow || !wrist) return;

    const upper = resolveBone(bones, isLeft
      ? ["mixamorig:LeftArm", "LeftArm", "mixamorigLeftArm"]
      : ["mixamorig:RightArm", "RightArm", "mixamorigRightArm"]);

    const lower = resolveBone(bones, isLeft
      ? ["mixamorig:LeftForeArm", "LeftForeArm", "mixamorigLeftForeArm"]
      : ["mixamorig:RightForeArm", "RightForeArm", "mixamorigRightForeArm"]);

    const hand = resolveBone(bones, isLeft
      ? ["mixamorig:LeftHand", "LeftHand", "mixamorigLeftHand"]
      : ["mixamorig:RightHand", "RightHand", "mixamorigRightHand"]);

    const upperDir = elbow.clone().sub(shoulder).normalize();
    const lowerDir = wrist.clone().sub(elbow).normalize();

    slerpBoneToDir(upper, upperDir, 0.55);
    slerpBoneToDir(lower, lowerDir, 0.7);

    if (hand) {
      const indexBase = getHandVec(frame, side, 5);
      const pinkyBase = getHandVec(frame, side, 17);
      if (indexBase && pinkyBase) {
        const palmAcross = pinkyBase.clone().sub(indexBase).normalize();
        const palmForward = lowerDir.clone().cross(palmAcross).normalize();
        const m = new THREE.Matrix4().lookAt(
          new THREE.Vector3(0, 0, 0),
          palmForward,
          new THREE.Vector3(0, 1, 0)
        );
        const q = new THREE.Quaternion().setFromRotationMatrix(m);
        hand.quaternion.slerp(q, 0.35);
      }
    }
  }

  function curlFinger(frame, bones, side, chainNames, lm0, lm1, lm2, lm3) {
    const isLeft = side === "left";
    const p0 = getHandVec(frame, side, lm0);
    const p1 = getHandVec(frame, side, lm1);
    const p2 = getHandVec(frame, side, lm2);
    const p3 = getHandVec(frame, side, lm3);
    if (!p0 || !p1 || !p2 || !p3) return;

    const a = p1.distanceTo(p0);
    const b = p2.distanceTo(p1);
    const c = p3.distanceTo(p2);
    const straight = a + b + c;
    const direct = p3.distanceTo(p0);
    const curl = clamp(1 - direct / Math.max(straight, 1e-5), 0, 1);

    for (const nameSet of chainNames) {
      const bone = resolveBone(bones, isLeft ? nameSet.left : nameSet.right);
      if (bone) {
        bone.rotation.x = lerpAngle(bone.rotation.x, -curl * 1.2, 0.45);
      }
    }
  }

  function lerpAngle(a, b, t) {
    return a + (b - a) * t;
  }

  function solveFingers(frame, bones, side) {
    const defs = [
      {
        lm: [1, 2, 3, 4],
        chain: [
          { left: ["mixamorig:LeftHandThumb1", "LeftHandThumb1"], right: ["mixamorig:RightHandThumb1", "RightHandThumb1"] },
          { left: ["mixamorig:LeftHandThumb2", "LeftHandThumb2"], right: ["mixamorig:RightHandThumb2", "RightHandThumb2"] },
          { left: ["mixamorig:LeftHandThumb3", "LeftHandThumb3"], right: ["mixamorig:RightHandThumb3", "RightHandThumb3"] }
        ]
      },
      {
        lm: [5, 6, 7, 8],
        chain: [
          { left: ["mixamorig:LeftHandIndex1", "LeftHandIndex1"], right: ["mixamorig:RightHandIndex1", "RightHandIndex1"] },
          { left: ["mixamorig:LeftHandIndex2", "LeftHandIndex2"], right: ["mixamorig:RightHandIndex2", "RightHandIndex2"] },
          { left: ["mixamorig:LeftHandIndex3", "LeftHandIndex3"], right: ["mixamorig:RightHandIndex3", "RightHandIndex3"] }
        ]
      },
      {
        lm: [9, 10, 11, 12],
        chain: [
          { left: ["mixamorig:LeftHandMiddle1", "LeftHandMiddle1"], right: ["mixamorig:RightHandMiddle1", "RightHandMiddle1"] },
          { left: ["mixamorig:LeftHandMiddle2", "LeftHandMiddle2"], right: ["mixamorig:RightHandMiddle2", "RightHandMiddle2"] },
          { left: ["mixamorig:LeftHandMiddle3", "LeftHandMiddle3"], right: ["mixamorig:RightHandMiddle3", "RightHandMiddle3"] }
        ]
      },
      {
        lm: [13, 14, 15, 16],
        chain: [
          { left: ["mixamorig:LeftHandRing1", "LeftHandRing1"], right: ["mixamorig:RightHandRing1", "RightHandRing1"] },
          { left: ["mixamorig:LeftHandRing2", "LeftHandRing2"], right: ["mixamorig:RightHandRing2", "RightHandRing2"] },
          { left: ["mixamorig:LeftHandRing3", "LeftHandRing3"], right: ["mixamorig:RightHandRing3", "RightHandRing3"] }
        ]
      },
      {
        lm: [17, 18, 19, 20],
        chain: [
          { left: ["mixamorig:LeftHandPinky1", "LeftHandPinky1"], right: ["mixamorig:RightHandPinky1", "RightHandPinky1"] },
          { left: ["mixamorig:LeftHandPinky2", "LeftHandPinky2"], right: ["mixamorig:RightHandPinky2", "RightHandPinky2"] },
          { left: ["mixamorig:LeftHandPinky3", "LeftHandPinky3"], right: ["mixamorig:RightHandPinky3", "RightHandPinky3"] }
        ]
      }
    ];

    for (const def of defs) {
      curlFinger(frame, bones, side, def.chain, def.lm[0], def.lm[1], def.lm[2], def.lm[3]);
    }
  }

  function solveTorso(frame, bones) {
    const lShoulder = getPoseVec(frame, 11);
    const rShoulder = getPoseVec(frame, 12);
    const lHip = getPoseVec(frame, 23);
    const rHip = getPoseVec(frame, 24);
    const nose = getPoseVec(frame, 0);

    if (!lShoulder || !rShoulder) return;

    const spine = resolveBone(bones, ["mixamorig:Spine", "Spine", "mixamorigSpine"]);
    const spine1 = resolveBone(bones, ["mixamorig:Spine1", "Spine1", "mixamorigSpine1"]);
    const spine2 = resolveBone(bones, ["mixamorig:Spine2", "Spine2", "mixamorigSpine2"]);
    const neck = resolveBone(bones, ["mixamorig:Neck", "Neck", "mixamorigNeck"]);
    const head = resolveBone(bones, ["mixamorig:Head", "Head", "mixamorigHead"]);

    const shoulderMid = lShoulder.clone().add(rShoulder).multiplyScalar(0.5);
    const hipMid = (lHip && rHip) ? lHip.clone().add(rHip).multiplyScalar(0.5) : shoulderMid.clone().add(new THREE.Vector3(0, -0.4, 0));
    const upDir = shoulderMid.clone().sub(hipMid).normalize();

    slerpBoneToDir(spine, upDir, 0.18);
    slerpBoneToDir(spine1, upDir, 0.18);
    slerpBoneToDir(spine2, upDir, 0.18);

    if (nose) {
      const headDir = nose.clone().sub(shoulderMid).normalize();
      slerpBoneToDir(neck, headDir, 0.15);
      slerpBoneToDir(head, headDir, 0.22);
    }
  }

  function applyAvatarFrameSolved(frame, bones) {
    if (!frame || !bones) return;
    solveTorso(frame, bones);
    solveArm(frame, bones, "left");
    solveArm(frame, bones, "right");
    solveFingers(frame, bones, "left");
    solveFingers(frame, bones, "right");
  }

  global.AvatarSolver = {
    applyAvatarFrameSolved
  };
})(window);