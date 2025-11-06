(async () => {
  // model/ファイルの場所（model.json と同階層に web_model フォルダがある想定）
  const MODEL_URL = './web_model/model.json';
  const SCALER_URL = './scaler.json';
  const LABELS_URL = './labels.json';

  const startBtn = document.getElementById('startBtn');
  const predLabelEl = document.getElementById('predLabel');
  const logs = document.getElementById('logs');
  const dots = document.querySelectorAll('.dot');

  let model = null;
  let scaler = null;
  let labels = null;

  // 先にボタンイベントを登録（モデル読み込みに失敗してもボタンは反応するように）
  startBtn.addEventListener('click', async () => {
    startBtn.disabled = true;
    try {
      await startMotion();
    } catch (e) {
      console.error('startMotion error', e);
      logs.innerText = 'センサー開始に失敗しました。コンソールを確認してください。';
      startBtn.disabled = false;
    }
  });

  // TF モデル読み込み + 前処理パラメータ読み込み
  async function loadAssets() {
    logs.innerText = 'モデル読み込み中...';
    try {
      model = await tf.loadLayersModel(MODEL_URL);
      logs.innerText = 'モデル読み込み完了。scaler 読み込み...';
      scaler = await (await fetch(SCALER_URL)).json();
      labels = await (await fetch(LABELS_URL)).json();
      logs.innerText = 'モデル準備完了。';
      console.log('labels:', labels);
    } catch (err) {
      console.error('loadAssets error', err);
      logs.innerText = 'モデルまたは前処理パラメータの読み込みに失敗しました。コンソールを確認してください。';
    }
  }

  await loadAssets();

  // デバイスモーションのバッファ（window of samples）
  const SAMPLING_TARGET = 200; // WISDM の 200 サンプル（約10秒）
  let buffer = [];

  function processSample(sample) {
    // sample = {x,y,z}
    buffer.push(sample);
    if (buffer.length > SAMPLING_TARGET) {
      buffer.shift();
    }
    // 推論は窓が満たされたとき／あるいは一定ごとに行う
    if (buffer.length === SAMPLING_TARGET) {
      doPredictFromBuffer();
    }
  }

  function computeFeatures(buf) {
    // buf: [{x,y,z}, ...] length == SAMPLING_TARGET
    const xs = buf.map(s => s.x);
    const ys = buf.map(s => s.y);
    const zs = buf.map(s => s.z);

    const mean = arr => arr.reduce((a,b)=>a+b,0)/arr.length;
    const std = arr => {
      const m = mean(arr);
      return Math.sqrt(arr.reduce((s,v)=>s+(v-m)**2,0)/arr.length);
    };
    const rms = () => {
      let s = 0;
      for (let i=0;i<buf.length;i++){
        const v = Math.sqrt(xs[i]*xs[i] + ys[i]*ys[i] + zs[i]*zs[i]);
        s += v;
      }
      return s / buf.length;
    };

    return [
      mean(xs), mean(ys), mean(zs),
      std(xs), std(ys), std(zs),
      rms()
    ];
  }

  function standardize(features) {
    // scaler.json には mean と scale 配列がある前提
    const mean = scaler?.mean || [];
    const scale = scaler?.scale || [];
    const out = [];
    for (let i=0;i<features.length;i++){
      const s = (features[i] - (mean[i] || 0)) / (scale[i] || 1e-8);
      out.push(s);
    }
    return out;
  }

  let lastPredTime = 0;
  async function doPredictFromBuffer() {
    // モデルやスケーラーが未読み込みなら推論をスキップ
    if (!model || !scaler || !labels) {
      const now = Date.now();
      if (now - lastPredTime > 2000) {
        logs.innerText = 'モデルが未読み込みのため推論をスキップしています...';
        lastPredTime = now;
      }
      return;
    }

    // 推論の頻度を制御（ここでは約0.8秒に1回）
    const now = Date.now();
    if (now - lastPredTime < 800) return;
    lastPredTime = now;

    try {
      // 特徴量算出→正規化→TF入力
      const feats = computeFeatures(buffer);
      const scaled = standardize(feats);
      const input = tf.tensor2d([scaled]);
      const out = model.predict(input);
      const probs = await out.data();
      input.dispose();
      out.dispose();

      // 予測ラベル（確率最大）
      let maxIdx = 0;
      for (let i=1;i<probs.length;i++) if (probs[i] > probs[maxIdx]) maxIdx = i;
      const label = labels[maxIdx] || 'unknown';

      // ラベルを "5段階" にマッピング（例）
      const mapToFive = (lab) => {
        if (!lab) return 3;
        const l = lab.toLowerCase();
        if (l.includes('sit') || l.includes('stand')) return 1;
        if (l === 'walking') return 3;
        if (l === 'upstairs' || l === 'downstairs') return 3;
        if (l === 'jogging' || l === 'running') return 5;
        return 3;
      };

      const five = mapToFive(label);
      updateUI(label, five, probs[maxIdx]);
    } catch (e) {
      console.error('prediction error', e);
      logs.innerText = '推論中にエラーが発生しました。コンソールを確認してください。';
    }
  }

  function updateUI(label, five, confidence) {
    predLabelEl.innerText = `${label} （${(confidence*100).toFixed(0)}%）`;
    dots.forEach(d => {
      const i = Number(d.dataset.i) + 1;
      d.classList.toggle('on', i === five);
    });
  }

  // DeviceMotion イベントのセットアップ（iOS の場合は permission が必要）
  async function startMotion() {
    // iOS Safari の permission フローに対応
    if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
      try {
        const res = await DeviceMotionEvent.requestPermission();
        if (res !== 'granted') {
          alert('DeviceMotion permission が必要です。設定を許可してください。');
          startBtn.disabled = false;
          return;
        }
      } catch (e) {
        console.warn('permission request failed', e);
        startBtn.disabled = false;
        return;
      }
    }

    // イベント登録（重複登録を避ける）
    const handler = (ev) => {
      const a = ev.accelerationIncludingGravity || ev.acceleration;
      if (!a) return;
      processSample({ x: a.x || 0, y: a.y || 0, z: a.z || 0 });
    };
    window.addEventListener('devicemotion', handler, { passive: true });
    logs.innerText = 'センサー取得中...（スマホを持って歩いてください）';
  }

})();
