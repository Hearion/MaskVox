(() => {
  'use strict';

  const $ = (selector) => document.querySelector(selector);
  const state = { settings: {}, presets: [], devices: { inputs: [], outputs: [] }, engine: { state: 'idle', message: '' } };
  let api = null;

  const FORMATS = {
    pitch: (value) => (value === 0 ? '原调' : `${signed(value)} 半音`),
    formant: (value) => (value === 0 ? '不变' : `${value > 0 ? '更细' : '更粗'} ${Math.abs(value)}`),
    echo: percent,
    reverb: percent,
    monitor_volume: percent,
    gate_db: (value) => (value <= -60 ? '关闭' : `−${Math.abs(value)} dB`),
    delay_ms: (value) => `${value} ms`,
  };

  function signed(value) { return `${value > 0 ? '+' : '−'}${Math.abs(value)}`; }
  function percent(value) { return value === 0 ? '关' : `${Math.round(value * 100)}%`; }

  // ---------- Bridge to the desktop program ----------

  function connect() {
    return new Promise((resolve) => {
      if (window.pywebview && window.pywebview.api) return resolve(window.pywebview.api);
      window.addEventListener('pywebviewready', () => resolve(window.pywebview.api), { once: true });
      // Opened in a plain browser: run on sample data so the page can be designed.
      setTimeout(() => { if (!window.pywebview) resolve(createPreviewApi()); }, 400);
    });
  }

  async function call(method, ...args) {
    try {
      const result = await api[method](...args);
      if (result && result.ok) return result.data;
      toast((result && result.error) || '操作失败', true);
    } catch (error) {
      toast('无法连接到桌面程序，请重新启动 MaskVox', true);
    }
    return undefined;
  }

  // ---------- Rendering ----------

  function currentPreset() {
    return state.presets.find((preset) => preset.id === state.settings.preset) || state.presets[0];
  }

  function renderEffects() {
    const grid = $('#effect-grid');
    grid.textContent = '';
    for (const preset of state.presets) {
      const tile = document.createElement('button');
      tile.type = 'button';
      tile.className = 'effect';
      tile.setAttribute('role', 'radio');
      tile.dataset.id = preset.id;
      tile.style.setProperty('--hue', preset.hue);
      const icon = Object.assign(document.createElement('span'), { className: 'effect-icon', textContent: preset.icon });
      icon.setAttribute('aria-hidden', 'true');
      const label = document.createElement('span');
      label.append(
        Object.assign(document.createElement('span'), { className: 'effect-name', textContent: preset.name }),
        Object.assign(document.createElement('span'), { className: 'effect-hint', textContent: preset.hint }),
      );
      tile.append(icon, label);
      tile.addEventListener('click', () => selectPreset(preset.id));
      grid.append(tile);
    }
    markActiveEffect();
  }

  function markActiveEffect() {
    for (const tile of document.querySelectorAll('.effect')) {
      const active = tile.dataset.id === state.settings.preset;
      tile.setAttribute('aria-checked', String(active));
      tile.tabIndex = active ? 0 : -1;
    }
  }

  function renderSliders() {
    for (const slider of document.querySelectorAll('.slider')) {
      const input = slider.querySelector('input');
      const value = Number(state.settings[slider.dataset.param]);
      if (document.activeElement !== input || Number(input.value) !== value) input.value = value;
      paintSlider(slider, Number(input.value));
    }
  }

  function paintSlider(slider, value) {
    const input = slider.querySelector('input');
    const span = Number(input.max) - Number(input.min);
    input.style.setProperty('--fill', `${((value - Number(input.min)) / span) * 100}%`);
    slider.querySelector('output').textContent = FORMATS[slider.dataset.param](value);
  }

  function renderDevices() {
    const groups = { input_id: state.devices.inputs, output_id: state.devices.outputs, monitor_id: state.devices.outputs };
    for (const select of document.querySelectorAll('select[data-device]')) {
      const key = select.dataset.device;
      select.textContent = '';
      const items = groups[key] || [];
      if (!items.length) select.append(new Option('没有可用设备', ''));
      for (const device of items) select.append(new Option(device.name, String(device.id)));
      const selected = state.settings[key];
      select.value = selected === null || selected === undefined ? '' : String(selected);
      select.disabled = !items.length;
    }
    $('#monitor').checked = Boolean(state.settings.monitor);
    const cable = state.devices.cable_output_id;
    $('#output-hint').textContent = cable !== null && cable !== undefined && state.settings.output_id === cable
      ? '已输出到虚拟麦克风。聊天软件里把麦克风设为 CABLE Output。'
      : '选 CABLE Input 才能把变声送进聊天软件和游戏。';
  }

  function renderCable() {
    const chip = $('#cable-chip');
    chip.hidden = false;
    if (state.has_vb_cable) {
      chip.className = 'chip';
      chip.textContent = '虚拟麦克风已就绪';
      chip.title = '';
    } else {
      chip.className = 'chip warn';
      chip.textContent = state.vb_cable_installer_ready ? '安装虚拟麦克风' : '获取虚拟麦克风';
      chip.title = '聊天软件和游戏需要 VB-Cable 才能收到变声';
    }
  }

  function renderEngine() {
    const power = $('#power');
    const { state: engine, message } = state.engine;
    const running = engine === 'running';
    const loading = engine === 'loading';
    power.setAttribute('aria-pressed', String(running));
    power.setAttribute('aria-label', running ? '停止变声' : '开始变声');
    power.classList.toggle('loading', loading);
    power.disabled = loading;
    const preset = currentPreset();
    $('#power-title').textContent = running ? '正在变声' : loading ? '正在启动…' : '点击开始变声';
    $('#power-subtitle').textContent = preset ? `当前音效：${preset.name}` : '';
    $('#status-dot').className = `status-dot ${running ? 'running' : engine === 'error' ? 'error' : ''}`;
    $('#status-text').textContent = running ? '运行中' : loading ? '启动中' : engine === 'error' ? (message || '已停止') : '未启动';
    if (!running) {
      $('#status-detail').textContent = '';
      setLevels(-90, -90);
    }
  }

  function setLevels(inDb, outDb) {
    const width = (db) => `${Math.max(0, Math.min(1, (db + 60) / 60)) * 100}%`;
    $('#meter-in').style.width = width(inDb);
    $('#meter-out').style.width = width(outDb);
    document.documentElement.style.setProperty('--level', String(Math.max(0, Math.min(1, (outDb + 50) / 40))));
  }

  function renderAll() {
    renderEffects();
    renderSliders();
    renderDevices();
    renderCable();
    renderEngine();
    $('#guide').hidden = Boolean(state.settings.onboarding_dismissed);
    $('#version').textContent = state.version ? `v${state.version}` : '';
  }

  function applySettings(settings) {
    if (!settings) return;
    state.settings = settings;
    markActiveEffect();
    renderSliders();
    renderDevices();
    renderEngine();
  }

  // ---------- Actions ----------

  async function selectPreset(id) {
    if (id === state.settings.preset) return;
    applySettings(await call('select_preset', id));
  }

  const pending = new Map();
  function queueParam(name, value) {
    // Coalesce drags: send the newest value once the previous call returns.
    const busy = pending.has(name);
    pending.set(name, value);
    if (busy) return;
    (async () => {
      while (pending.has(name)) {
        const next = pending.get(name);
        const settings = await call('set_param', name, next);
        if (pending.get(name) === next) pending.delete(name);
        if (settings && !pending.size) applySettings(settings);
      }
    })();
  }

  async function togglePower() {
    if (state.engine.state === 'running') await call('stop');
    else await call('start');
  }

  async function onCableChip() {
    if (state.has_vb_cable) return;
    if (state.vb_cable_installer_ready) {
      if ((await call('install_vb_cable')) !== undefined) toast('安装程序已打开。装完后重启电脑，再回来刷新设备。');
    } else {
      await call('open_url', 'https://vb-audio.com/Cable/');
    }
  }

  function toggleDrawer(open) {
    $('#drawer').hidden = !open;
    $('#scrim').hidden = !open;
    if (open) $('#close-settings').focus();
    else $('#open-settings').focus();
  }

  function toast(message, isError) {
    const item = Object.assign(document.createElement('div'), { className: `toast${isError ? ' error' : ''}`, textContent: message });
    $('#toasts').append(item);
    setTimeout(() => item.remove(), isError ? 6000 : 3500);
  }

  // ---------- Events from the audio engine ----------

  window.onEngineEvent = ({ type, payload }) => {
    if (type === 'status') {
      state.engine = payload;
      renderEngine();
      if (payload.state === 'error' && payload.message) toast(payload.message, true);
    } else if (type === 'meter') {
      setLevels(payload.in_db, payload.out_db);
      const glitches = payload.underruns ? ` · 断续 ${payload.underruns} 次` : '';
      $('#status-detail').textContent = `延迟约 ${payload.latency_ms} ms · 处理 ${payload.process_ms.toFixed(1)} ms${glitches}`;
    } else if (type === 'warning') {
      toast(payload.message, true);
    }
  };

  function bind() {
    $('#power').addEventListener('click', togglePower);
    $('#cable-chip').addEventListener('click', onCableChip);
    $('#open-settings').addEventListener('click', () => toggleDrawer(true));
    $('#close-settings').addEventListener('click', () => toggleDrawer(false));
    $('#scrim').addEventListener('click', () => toggleDrawer(false));
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && !$('#drawer').hidden) toggleDrawer(false);
    });
    $('#reset-tune').addEventListener('click', async () => {
      applySettings(await call('select_preset', state.settings.preset));
    });
    $('#dismiss-guide').addEventListener('click', async () => {
      $('#guide').hidden = true;
      await call('dismiss_onboarding');
    });
    for (const slider of document.querySelectorAll('.slider')) {
      const input = slider.querySelector('input');
      // A slider that restarts the audio streams sends its value on release only.
      const onRelease = slider.dataset.commit === 'release';
      input.addEventListener('input', () => {
        paintSlider(slider, Number(input.value));
        if (!onRelease) queueParam(slider.dataset.param, Number(input.value));
      });
      if (onRelease) input.addEventListener('change', () => queueParam(slider.dataset.param, Number(input.value)));
    }
    for (const select of document.querySelectorAll('select[data-device]')) {
      select.addEventListener('change', async () => {
        const settings = await call('set_devices', { [select.dataset.device]: select.value === '' ? null : Number(select.value) });
        if (settings) applySettings(settings);
        else renderDevices();
      });
    }
    $('#monitor').addEventListener('change', async (event) => {
      const settings = await call('set_param', 'monitor', event.target.checked);
      if (settings) applySettings(settings);
      else renderDevices();
    });
    $('#refresh-devices').addEventListener('click', async () => {
      const result = await call('list_devices');
      if (!result) return;
      state.devices = result.devices;
      state.has_vb_cable = result.devices.has_vb_cable;
      applySettings(result.settings);
      renderCable();
      toast('设备列表已刷新');
    });
    $('#open-logs').addEventListener('click', () => call('open_logs'));
    $('#effect-grid').addEventListener('keydown', (event) => {
      const moves = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 4, ArrowUp: -4 };
      if (!(event.key in moves)) return;
      const tiles = [...document.querySelectorAll('.effect')];
      const next = tiles[tiles.indexOf(document.activeElement) + moves[event.key]];
      if (next) { event.preventDefault(); next.focus(); next.click(); }
    });
  }

  // ---------- Sample data for a plain browser ----------

  function createPreviewApi() {
    const names = [
      ['original', '原声', '不做处理', '🎙️', 210, 0, 0, 0, 0], ['woman', '女声', '男声变女声', '👩', 330, 9, 3, 0, 0],
      ['loli', '萝莉', '又细又甜', '🎀', 345, 12, 5, 0, 0], ['boy', '正太', '稚气少年', '🧒', 190, 5, 2, 0, 0],
      ['man', '大叔', '女声变男声', '🧔', 28, -5, -2.5, 0, 0], ['giant', '巨人', '低沉厚重', '🗿', 160, -9, -5, 0, 0.15],
      ['demon', '恶魔', '嘶哑阴森', '😈', 0, -7, -4, 0, 0.25], ['chipmunk', '花栗鼠', '快放的磁带', '🐿️', 40, 12, 9, 0, 0],
      ['robot', '机器人', '没有语调', '🤖', 200, 0, 0, 0, 0], ['alien', '外星人', '金属颤音', '👽', 130, 3, 0, 0, 0],
      ['ghost', '幽灵', '气声加回响', '👻', 265, 0, -1, 0, 0.45], ['tune', '电音', '音高吸附到音阶', '🎛️', 285, 0, 0, 0, 0],
      ['phone', '电话', '老式听筒', '☎️', 50, 0, 0, 0, 0], ['radio', '对讲机', '失真加窄带', '📻', 15, 0, 0, 0, 0],
      ['cave', '山洞', '回声阵阵', '🏔️', 175, 0, 0, 0.4, 0.5], ['underwater', '水下', '闷闷的咕噜声', '🌊', 220, 0, 0, 0, 0.2],
    ];
    const presets = names.map(([id, name, hint, icon, hue]) => ({ id, name, hint, icon, hue }));
    const settings = {
      input_id: 1, output_id: 4, monitor_id: 3, preset: 'woman', pitch: 9, formant: 3, echo: 0, reverb: 0,
      gate_db: -50, delay_ms: 30, monitor: false, monitor_volume: 0.6, onboarding_dismissed: false,
    };
    const devices = {
      inputs: [{ id: 1, name: '麦克风 (HyperX Cloud II)' }, { id: 2, name: '麦克风阵列 (Realtek Audio)' }],
      outputs: [{ id: 3, name: '耳机 (HyperX Cloud II)' }, { id: 4, name: 'CABLE Input (VB-Audio Virtual Cable)' }],
      cable_output_id: 4, has_vb_cable: true,
    };
    let timer = null;
    const ok = (data) => Promise.resolve({ ok: true, data });
    const emit = (type, payload) => window.onEngineEvent({ type, payload });
    return {
      get_state: () => ok({
        settings: { ...settings }, presets, devices, engine: { state: 'idle', message: '' },
        has_vb_cable: true, vb_cable_installer_ready: true, version: '2.0.0',
      }),
      select_preset: (id) => {
        const row = names.find((item) => item[0] === id);
        Object.assign(settings, { preset: id, pitch: row[5], formant: row[6], echo: row[7], reverb: row[8] });
        return ok({ ...settings });
      },
      set_param: (name, value) => { settings[name] = value; return ok({ ...settings }); },
      set_devices: (selection) => { Object.assign(settings, selection); return ok({ ...settings }); },
      list_devices: () => ok({ devices, settings: { ...settings } }),
      dismiss_onboarding: () => { settings.onboarding_dismissed = true; return ok({ ...settings }); },
      start: () => {
        emit('status', { state: 'loading', message: '' });
        setTimeout(() => {
          emit('status', { state: 'running', message: '' });
          timer = setInterval(() => {
            const level = -34 + 22 * Math.abs(Math.sin(Date.now() / 260)) * Math.random();
            emit('meter', { in_db: level, out_db: level + 2, process_ms: 0.4, latency_ms: 118, underruns: 0 });
          }, 100);
        }, 500);
        return ok(null);
      },
      stop: () => { clearInterval(timer); emit('status', { state: 'idle', message: '' }); return ok(null); },
      install_vb_cable: () => ok(null),
      open_url: () => ok(null),
      open_logs: () => ok(null),
    };
  }

  async function start() {
    bind();
    api = await connect();
    const initial = await call('get_state');
    if (!initial) return;
    Object.assign(state, initial);
    renderAll();
  }

  document.addEventListener('DOMContentLoaded', start, { once: true });
})();
