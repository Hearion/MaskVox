(() => {
  'use strict';

  const app = window.MaskVox;
  const COUNT = 20;
  const BAR_COUNT = 14;
  const MIN_DB = -66;
  const clamp = (value) => Math.min(1, Math.max(0, value));
  const normalize = (db) => clamp((db - MIN_DB) / -MIN_DB);

  function meterBars(container, output) {
    const hueRange = output ? [186, 262] : [258, 278];
    const bars = Array.from({ length: BAR_COUNT }, (_, index) => {
      const bar = document.createElement('i');
      const hue = hueRange[0] + (hueRange[1] - hueRange[0]) * (index / (BAR_COUNT - 1));
      bar.style.setProperty('--bar-color', `hsl(${hue} 82% 66%)`);
      container.append(bar);
      return bar;
    });
    let oldCount = -1;
    return (level) => {
      const count = Math.round(level * BAR_COUNT);
      if (count === oldCount) return;
      for (let index = 0; index < bars.length; index++) bars[index].dataset.on = String(index < count);
      oldCount = count;
    };
  }

  app.createMeters = ({ canvas, power, inputBars, outputBars, inputText, outputText }) => {
    const context = canvas.getContext('2d');
    const reduced = matchMedia('(prefers-reduced-motion: reduce)');
    const updateIn = meterBars(inputBars, false);
    const updateOut = meterBars(outputBars, true);
    const history = new Float32Array(COUNT + 1);
    let width = 1;
    let height = 1;
    let running = false;
    let receivedAt = 0;
    let inputLevel = 0;
    let outputLevel = 0;
    let shownInput = 0;
    let shownOutput = 0;
    let lastFrame = 0;
    let frame = 0;
    const displayDb = (value) => value <= -84 ? '— dB' : `${Math.round(value).toString().replace('-', '−')} dB`;

    function draw(now) {
      if (!context) return;
      context.clearRect(0, 0, width, height);
      const midpoint = height / 2;
      const gradient = context.createLinearGradient(0, 0, width, 0);
      gradient.addColorStop(0, '#a182ff');
      gradient.addColorStop(.62, '#8b71fd');
      gradient.addColorStop(1, '#42cded');
      if (!running) {
        context.fillStyle = '#665786';
        context.fillRect(13, Math.floor(midpoint), Math.max(0, width - 26), 1);
        return;
      }
      context.fillStyle = '#544973';
      context.fillRect(13, Math.floor(midpoint), Math.max(0, width - 26), 1);
      const gap = Math.max(1, (width - 26) / COUNT);
      const barWidth = Math.max(1, Math.min(7, gap * .54));
      const blend = reduced.matches ? 1 : clamp((now - receivedAt) / 100);
      context.fillStyle = gradient;
      for (let index = 0; index < COUNT; index++) {
        const old = history[index];
        const next = history[index + 1];
        const level = old + (next - old) * blend;
        const barHeight = 2 + level * level * (height - 14);
        const x = 13 + index * gap + (gap - barWidth) / 2;
        context.globalAlpha = .26 + .7 * ((index + 1) / COUNT);
        context.beginPath();
        context.roundRect(x, midpoint - barHeight / 2, barWidth, barHeight, barWidth / 2);
        context.fill();
      }
      context.globalAlpha = 1;
    }

    function animate(now) {
      frame = 0;
      if (!running) return;
      const blend = reduced.matches ? 1 : 1 - Math.exp(-Math.min(60, now - lastFrame) / 72);
      lastFrame = now;
      shownInput += (inputLevel - shownInput) * blend;
      shownOutput += (outputLevel - shownOutput) * blend;
      updateIn(shownInput);
      updateOut(shownOutput);
      power.style.setProperty('--signal', reduced.matches ? '.2' : shownOutput.toFixed(3));
      draw(now);
      if (!reduced.matches) frame = requestAnimationFrame(animate);
    }

    const queueFrame = () => {
      if (frame) return;
      lastFrame = performance.now();
      frame = requestAnimationFrame(animate);
    };
    new ResizeObserver(() => {
      const { width: measuredWidth, height: measuredHeight } = canvas.getBoundingClientRect();
      width = measuredWidth;
      height = measuredHeight;
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      context?.setTransform(scale, 0, 0, scale, 0, 0);
      draw(performance.now());
    }).observe(canvas);

    updateIn(0);
    updateOut(0);
    return {
      setRunning(value) {
        if (running === value) return;
        running = value;
        if (value) {
          queueFrame();
          return;
        }
        cancelAnimationFrame(frame);
        frame = 0;
        shownInput = shownOutput = inputLevel = outputLevel = 0;
        history.fill(0);
        updateIn(0);
        updateOut(0);
        inputText.textContent = outputText.textContent = '— dB';
        draw(performance.now());
      },
      sample(payload) {
        if (!running) return;
        const inDb = Number.isFinite(payload.in_db) ? payload.in_db : -90;
        const outDb = Number.isFinite(payload.out_db) ? payload.out_db : -90;
        inputLevel = normalize(inDb);
        outputLevel = normalize(outDb);
        inputText.textContent = displayDb(inDb);
        outputText.textContent = displayDb(outDb);
        history.copyWithin(0, 1);
        history[COUNT] = outputLevel;
        receivedAt = performance.now();
        queueFrame();
      },
    };
  };
})();
