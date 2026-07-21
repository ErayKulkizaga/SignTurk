(function (global) {
  function cloneFrame(frame) {
    return JSON.parse(JSON.stringify(frame || {}));
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function lerpPoint(pa, pb, t) {
    if (!pa && !pb) return null;
    if (!pa) return pb.slice();
    if (!pb) return pa.slice();
    return [
      lerp(pa[0], pb[0], t),
      lerp(pa[1], pb[1], t),
      lerp(pa[2] || 0, pb[2] || 0, t)
    ];
  }

  function smoothArray(points, alpha) {
    if (!points || !points.length) return points;
    const out = points.map(p => p ? p.slice() : null);
    for (let i = 1; i < out.length; i++) {
      if (!out[i] || !out[i - 1]) continue;
      out[i][0] = lerp(out[i][0], out[i - 1][0], alpha);
      out[i][1] = lerp(out[i][1], out[i - 1][1], alpha);
      out[i][2] = lerp(out[i][2], out[i - 1][2], alpha);
    }
    return out;
  }

  function fillMissingHand(frames, key) {
    let last = null;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i][key] && frames[i][key].length) last = frames[i][key];
      else if (last) frames[i][key] = JSON.parse(JSON.stringify(last));
    }
    let next = null;
    for (let i = frames.length - 1; i >= 0; i--) {
      if (frames[i][key] && frames[i][key].length) next = frames[i][key];
      else if (next) frames[i][key] = JSON.parse(JSON.stringify(next));
    }
  }

  function smoothHand(frames, key, alpha) {
    if (!frames.length) return;
    const count = 21;
    for (let lm = 0; lm < count; lm++) {
      const track = frames.map(f => (f[key] && f[key][lm]) ? f[key][lm] : null);
      const smoothed = smoothArray(track, alpha);
      for (let i = 0; i < frames.length; i++) {
        if (!frames[i][key]) frames[i][key] = [];
        if (smoothed[i]) frames[i][key][lm] = smoothed[i];
      }
    }
  }

  function smoothPose(frames, alpha) {
    const keys = ["0", "11", "12", "13", "14", "15", "16", "23", "24"];
    for (const k of keys) {
      const track = frames.map(f => f.pose && f.pose[k] ? f.pose[k] : null);
      const smoothed = smoothArray(track, alpha);
      for (let i = 0; i < frames.length; i++) {
        if (!frames[i].pose) frames[i].pose = {};
        if (smoothed[i]) frames[i].pose[k] = smoothed[i];
      }
    }
  }

  function resampleFrames(frames, targetCount) {
    if (!frames || !frames.length) return [];
    if (frames.length === targetCount) return frames.map(cloneFrame);

    const out = [];
    const maxSrc = frames.length - 1;
    for (let i = 0; i < targetCount; i++) {
      const u = (i / Math.max(1, targetCount - 1)) * maxSrc;
      const a = Math.floor(u);
      const b = Math.min(maxSrc, a + 1);
      const t = u - a;

      const fa = frames[a];
      const fb = frames[b];
      const nf = { pose: {}, left_hand: [], right_hand: [] };

      const poseKeys = ["0", "11", "12", "13", "14", "15", "16", "23", "24"];
      for (const k of poseKeys) {
        const pa = fa.pose && fa.pose[k] ? fa.pose[k] : null;
        const pb = fb.pose && fb.pose[k] ? fb.pose[k] : null;
        const v = lerpPoint(pa, pb, t);
        if (v) nf.pose[k] = v;
      }

      for (let j = 0; j < 21; j++) {
        nf.left_hand[j] = lerpPoint(fa.left_hand && fa.left_hand[j], fb.left_hand && fb.left_hand[j], t);
        nf.right_hand[j] = lerpPoint(fa.right_hand && fa.right_hand[j], fb.right_hand && fb.right_hand[j], t);
      }

      out.push(nf);
    }
    return out;
  }

  function preprocessLandmarkData(data) {
    if (!data || !data.frames || !data.frames.length) return data;

    const result = JSON.parse(JSON.stringify(data));
    result.frames = resampleFrames(result.frames, Math.max(36, result.frames.length));

    fillMissingHand(result.frames, "left_hand");
    fillMissingHand(result.frames, "right_hand");

    smoothPose(result.frames, 0.35);
    smoothHand(result.frames, "left_hand", 0.28);
    smoothHand(result.frames, "right_hand", 0.28);

    result.frame_count = result.frames.length;
    result.processed = true;
    return result;
  }

  global.MotionPreprocess = {
    preprocessLandmarkData
  };
})(window);