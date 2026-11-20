(() => {
  'use strict';

  const app = window.MaskVox;
  const CABLE_URL = 'https://vb-audio.com/Cable/';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const SLIDERS = {
    pitch: ['pitch', (value) => `${value > 0 ? '+' : ''}${value} 半音`.replace('-', '−')],
    index_rate: ['index-rate', (value) => Number(value).toFixed(2)],
    gate_db: ['gate-db', (value) => value === -60 ? '关闭' : `−${Math.abs(value)} dB`],
    monitor_volume: ['monitor-volume', (value) => `${Math.round(value * 100)}%`],
  };
  const DEVICE_FIELDS = { input_id: 'input-device', output_id: 'output-device', monitor_id: 'monitor-device' };
  const IMPORT_STAGE = { loading: '正在读取', exporting: '正在转换', verifying: '正在校验' };
  const $ = (id) => document.getElementById(id);
  const click = (id, callback) => $(id).addEventListener('click', callback);
  const basename = (path) => String(path || '').split(/[\\/]/).at(-1);
  const engineLabels = { idle: '待机', loading: '正在加载', running: '运行中', error: '运行出错' };

  function icon(name) {
    const svg = document.createElementNS(SVG_NS, 'svg');
    const use = document.createElementNS(SVG_NS, 'use');
    use.setAttribute('href', `#i-${name}`);
    svg.setAttribute('aria-hidden', 'true');
    svg.append(use);
    return svg;
  }

  function notify(message, kind = 'info') {
    if (!message) return;
    const stack = $('toasts');
    if ([...stack.querySelectorAll('p')].some((item) => item.textContent === message)) return;
    const toast = document.createElement('div');
    const content = document.createElement('p');
    const close = document.createElement('button');
    toast.className = 'toast';
    toast.dataset.kind = kind;
    content.textContent = message;
    close.className = 'icon-button';
    close.type = 'button';
    close.setAttribute('aria-label', '关闭提示');
    close.append(icon('close'));
    close.addEventListener('click', () => toast.remove());
    toast.append(icon(kind === 'error' || kind === 'warning' ? 'alert' : 'info'), content, close);
    stack.append(toast);
    if (kind !== 'error') setTimeout(() => toast.remove(), 3000);
  }

  function configureDialog() {
    const dialog = $('modal');
    click('modal-close', () => dialog.close());
    click('modal-cancel', () => dialog.close());
    $('modal-form').addEventListener('submit', (event) => {
      event.preventDefault();
      const input = $('modal-input');
      if (input.required && !input.value.trim()) {
        input.setCustomValidity('请输入音色名称');
        input.reportValidity();
        return;
      }
      dialog.close('confirm');
    });
    $('modal-input').addEventListener('input', (event) => event.currentTarget.setCustomValidity(''));

    return ({ title, message, value, confirm = '确认', danger = false }) => {
      if (dialog.open) return Promise.resolve(null);
      const input = $('modal-input');
      $('modal-title').textContent = title;
      $('modal-message').textContent = message;
      $('modal-field').hidden = value === undefined;
      input.value = value ?? '';
      input.required = value !== undefined;
      input.setCustomValidity('');
      $('modal-confirm').textContent = confirm;
      $('modal-confirm').classList.toggle('danger-button', danger);
      dialog.returnValue = '';
      return new Promise((resolve) => {
        dialog.addEventListener('close', () => {
          resolve(dialog.returnValue === 'confirm' ? input.value.trim() : null);
        }, { once: true });
        dialog.showModal();
        if (input.required) {
          input.focus();
          input.select();
        } else {
          $('modal-cancel').focus();
        }
      });
    };
  }

  function setSliderDisplay(input, value, format) {
    input.value = String(value);
    const fill = `${100 * (Number(value) - Number(input.min)) / (Number(input.max) - Number(input.min))}%`;
    input.parentElement.style.setProperty('--fill', fill);
    const text = format(value);
    $(`${input.id}-value`).textContent = text;
    input.nextElementSibling.textContent = text;
    input.setAttribute('aria-valuetext', text);
  }

  function setOptions(id, list, selected, emptyLabel) {
    const select = $(id);
    const chosen = list.find((item) => item.id === selected);
    const entries = chosen ? list : [{ id: null, name: list.length ? '请选择设备' : emptyLabel }, ...list];
    const signature = JSON.stringify(entries);
    if (select.dataset.options !== signature) {
      const options = entries.map(({ id, name }) => new Option(name, id ?? ''));
      if (!chosen) options[0].disabled = true;
      select.replaceChildren(...options);
      select.dataset.options = signature;
    }
    select.disabled = !list.length;
    select.value = chosen ? String(selected) : '';
    select.title = chosen?.name ?? '';
  }

  app.start = async () => {
    const store = app.createStore({
      connected: false, settings: { ...app.defaults }, voices: [], devices: null, engine: { state: 'idle' },
      has_vb_cable: false, base_models_ready: false, converter_installed: false, gpu_adapters: [],
      logs_path: '', version: '', busy: null, indexUnavailable: false,
    });
    const pendingSliders = new Set();
    const freshVoices = new Set();
    const jobCards = new Map();
    const showDialog = configureDialog();
    const meters = app.createMeters({
      canvas: $('waveform'), power: $('console-main'), inputBars: $('in-bars'), outputBars: $('out-bars'),
      inputText: $('in-db'), outputText: $('out-db'),
    });
    const settingsDialog = $('settings');
    const statusStats = $('status-stats');
    const voiceLabel = $('current-voice');
    let bridge;
    let refreshing = null;
    let refreshAgain = false;
    let checkingConverter = false;
    let openingPicker = false;
    let togglingMonitor = false;
    let steadyMeters = 0;
    let statusRevision = 0;

    async function safeCall(method, ...args) {
      try {
        return await bridge.call(method, ...args);
      } catch (error) {
        if (error.code === 'converter_missing') promptConverter();
        else notify(error.message, 'error');
        return undefined;
      }
    }

    async function refreshState() {
      if (refreshing) {
        refreshAgain = true;
        return refreshing;
      }
      refreshing = (async () => {
        do {
          refreshAgain = false;
          const revision = statusRevision;
          const state = await safeCall('get_state');
          if (state) {
            const engine = statusRevision === revision ? state.engine : store.read().engine;
            store.update({ ...state, engine, connected: true });
          }
        } while (refreshAgain);
      })();
      try {
        await refreshing;
      } finally {
        refreshing = null;
      }
    }

    function applySettings(values, keys) {
      if (!values) return;
      const patch = Object.fromEntries(keys.filter((key) => Object.hasOwn(values, key)).map((key) => [key, values[key]]));
      store.update({ settings: { ...store.read().settings, ...patch } });
    }

    async function withBusy(label, callback) {
      if (!store.read().connected || store.read().busy) return;
      store.update({ busy: label });
      try {
        await callback();
      } finally {
        store.update({ busy: null });
      }
    }

    const viewLibrary = app.createVoiceLibrary($('voice-list'), { voice: $('voice-template') }, {
      select: (voice) => {
        if (voice.id === store.read().settings.selected_voice_id) return;
        void withBusy('正在切换…', async () => {
          const settings = await safeCall('select_voice', voice.id);
          if (settings) applySettings(settings, ['selected_voice_id']);
          else if (bridge.preview) await refreshState();
        });
      },
      rename: async (voice) => {
        const name = await showDialog({ title: '重命名音色', message: '新名称会显示在音色库中。', value: voice.name, confirm: '保存' });
        if (name === null || name === voice.name) return;
        await safeCall('rename_voice', voice.id, name);
        await refreshState();
      },
      export: async (voice) => {
        const path = await safeCall('export_voice', voice.id);
        if (path) notify(`${bridge.preview ? '已模拟导出' : '音色包已导出'} · ${basename(path)}`);
      },
      delete: async (voice) => {
        const confirmed = await showDialog({ title: '删除音色', message: `要从音色库删除「${voice.name}」吗？`, confirm: '删除', danger: true });
        if (confirmed === null) return;
        await safeCall('delete_voice', voice.id);
        await refreshState();
      },
    });

    function render(state) {
      const { settings, engine, devices, busy, connected } = state;
      const voice = state.voices.find((item) => item.id === settings.selected_voice_id);
      const phase = busy ? 'loading' : engine.state;
      const waiting = phase === 'loading';
      const running = phase === 'running';
      $('power').disabled = !connected || waiting;
      $('power').setAttribute('aria-label', running ? '停止变声' : '启动变声');
      $('power').setAttribute('aria-pressed', String(running));
      $('power').title = running ? '停止变声' : '启动变声';
      $('console-main').dataset.state = phase;
      $('console-main').setAttribute('aria-busy', String(waiting));
      $('status-dot').dataset.state = phase;
      $('status-text').textContent = connected ? engineLabels[phase] : phase === 'error' ? '连接失败' : '正在连接';
      const captions = { idle: voice ? '点击启动变声' : '导入音色后即可变声', loading: '正在加载音色…', running: '点击停止变声', error: '运行出错，点击重试' };
      $('power-caption').textContent = busy || (waiting && engine.message) || (connected ? captions[phase] : phase === 'error' ? '无法连接桌面程序' : '正在连接…');
      $('power-caption').title = engine.message || '';
      meters.setRunning(engine.state === 'running');
      if (engine.state !== 'running') {
        updateLatency({});
        $('performance-flag').hidden = true;
      }

      const backend = engine.backend;
      $('backend').dataset.backend = backend || '';
      const backendText = backend === 'DirectML' ? '显卡加速 · DirectML' : backend === 'CPU' ? 'CPU 推理' : '推理后端待启动';
      $('backend').querySelector('span').textContent = bridge?.preview ? `界面预览 · ${backendText}` : backendText;
      if (voice) {
        if (voiceLabel.dataset.key !== `${voice.id}\n${voice.name}`) {
          voiceLabel.dataset.key = `${voice.id}\n${voice.name}`;
          const name = document.createElement('strong');
          name.textContent = voice.name;
          name.title = voice.name;
          const spec = document.createElement('span');
          spec.className = 'voice-spec';
          spec.textContent = app.voiceSpec(voice);
          voiceLabel.replaceChildren('当前音色', name, spec);
        }
      } else {
        delete voiceLabel.dataset.key;
        voiceLabel.textContent = '未选择音色';
      }

      $('cable-guide').hidden = !connected || state.has_vb_cable;
      $('assets-guide').hidden = !connected || state.base_models_ready;
      $('empty-library').hidden = !connected || state.voices.length > 0 || jobCards.size > 0;
      $('onboarding').hidden = !connected || settings.onboarding_dismissed;
      $('import-voices').disabled = !connected || openingPicker;
      $('voice-count').textContent = String(state.voices.length);
      viewLibrary(state.voices, settings.selected_voice_id, waiting, freshVoices);

      for (const [parameter, [id, format]] of Object.entries(SLIDERS)) {
        if (!pendingSliders.has(parameter)) setSliderDisplay($(id), settings[parameter], format);
        $(id).disabled = !connected;
      }
      const indexDisabled = !voice?.has_index || state.indexUnavailable;
      $('index-rate').disabled = !connected || indexDisabled;
      $('index-note').hidden = !indexDisabled;
      $('index-note').textContent = state.indexUnavailable ? '当前音色的索引无法使用' : '当前音色无检索索引';
      for (const button of document.querySelectorAll('[data-latency]')) {
        button.setAttribute('aria-pressed', String(button.dataset.latency === settings.latency_mode));
        button.disabled = !connected || waiting;
      }
      $('monitor').checked = settings.monitor;
      $('monitor').disabled = !connected || togglingMonitor;
      $('monitor-tip').hidden = !settings.monitor;
      $('monitor-volume-control').hidden = !settings.monitor;

      const outputs = devices?.outputs ?? [];
      setOptions('input-device', devices?.inputs ?? [], settings.input_id, '未发现麦克风');
      setOptions('output-device', outputs, settings.output_id, '未发现输出设备');
      setOptions('monitor-device', outputs.filter((device) => device.id !== settings.output_id), settings.monitor_id, '未发现监听设备');
      for (const id of Object.values(DEVICE_FIELDS)) $(id).disabled ||= waiting;
      $('refresh-devices').disabled = !connected || waiting || running;
      $('refresh-devices').title = running ? '停止变声后刷新设备' : '刷新设备';
      const adapters = state.gpu_adapters.length ? state.gpu_adapters : [{ id: 0, name: '系统默认显卡' }];
      setOptions('gpu-index', [...adapters, { id: -1, name: 'CPU（兼容模式）' }], settings.gpu_index, '未发现显卡');
      $('gpu-index').disabled = !connected || waiting;
      const checks = [
        ['base-status', state.base_models_ready, '已就绪', '缺少文件'],
        ['converter-status', state.converter_installed, '已安装', '未安装'],
      ];
      for (const [id, ready, yes, no] of checks) {
        $(id).dataset.ready = connected ? String(ready) : '';
        $(id).textContent = connected ? ready ? yes : no : '正在检查';
      }
      $('logs-path').textContent = state.logs_path;
      $('version').textContent = state.version;
    }

    function updateLatency(payload) {
      const milliseconds = (value) => Number.isFinite(value) ? Math.round(value) : '—';
      const dot = document.createElement('span');
      dot.textContent = '·';
      statusStats.replaceChildren(`推理 ${milliseconds(payload.infer_ms)} ms`, dot, `总延迟约 ${milliseconds(payload.latency_ms)} ms`);
      statusStats.title = payload.underruns > 0 ? `输出欠载 ${payload.underruns} 次` : '';
    }

    async function promptConverter() {
      if (checkingConverter) return;
      checkingConverter = true;
      try {
        const open = await showDialog({
          title: '需要安装转换组件',
          message: '导入 .pth 音色需要转换组件（约 1 GB）。安装后重新导入即可。\n.maskvox 音色包可直接导入。',
          confirm: '查看安装方法',
        });
        if (open !== null) await safeCall('open_install_guide');
      } finally {
        checkingConverter = false;
      }
    }

    function progress(payload) {
      const { job_id: id, stage, file, voice_id: voiceId, message, code } = payload;
      if (!id) return;
      if (IMPORT_STAGE[stage]) {
        if (!jobCards.has(id)) {
          const card = $('job-template').content.firstElementChild.cloneNode(true);
          const cancel = card.querySelector('.cancel-job');
          cancel.addEventListener('click', async () => {
            cancel.disabled = true;
            await safeCall('cancel_import', id);
            if (card.isConnected) cancel.disabled = false;
          });
          $('import-jobs').append(card);
          jobCards.set(id, card);
        }
        const card = jobCards.get(id);
        const name = basename(file) || '正在导入音色';
        card.querySelector('.job-file').textContent = name;
        card.querySelector('.job-file').title = name;
        card.querySelector('.job-stage').textContent = IMPORT_STAGE[stage];
        card.querySelector('progress').value = Math.max(0, Math.min(1, payload.progress ?? 0));
      } else {
        jobCards.get(id)?.remove();
        jobCards.delete(id);
        if (stage === 'error') {
          if (code === 'converter_missing') void promptConverter();
          else notify(message || '导入失败，请查看日志', 'error');
        } else if (stage === 'cancelled') {
          notify(message || '已取消导入');
        } else if (stage === 'done') {
          if (voiceId) freshVoices.add(voiceId);
          notify(message || '音色已导入', payload.partial ? 'warning' : 'success');
          void refreshState();
        }
      }
      $('empty-library').hidden = store.read().voices.length > 0 || jobCards.size > 0;
    }

    function emit(type, payload = {}) {
      switch (type) {
        case 'meter':
          if (store.read().engine.state === 'running') {
            meters.sample(payload);
            updateLatency(payload);
            if (!$('performance-flag').hidden) {
              const mode = store.read().settings.latency_mode;
              const blockMs = { low: 120, balanced: 200, high: 350 }[mode];
              steadyMeters = payload.infer_ms < blockMs * .8 ? steadyMeters + 1 : 0;
              if (steadyMeters >= 10) $('performance-flag').hidden = true;
            }
          }
          break;
        case 'status':
          statusRevision++;
          store.update({ engine: payload, ...(payload.state === 'loading' ? { indexUnavailable: false } : {}) });
          if (payload.state === 'error') notify(payload.message || '变声引擎发生错误', 'error');
          break;
        case 'warning':
          if (payload.code === 'performance') {
            steadyMeters = 0;
            $('performance-flag').hidden = false;
          }
          if (payload.code === 'index_invalid' || payload.code === 'index_missing') store.update({ indexUnavailable: true });
          notify(payload.message, 'warning');
          break;
        case 'import_progress':
          progress(payload);
          break;
        case 'voices_changed':
          void refreshState();
          break;
      }
    }

    window.onEngineEvent = (event, detail) => {
      if (!bridge) return;
      if (typeof event === 'string') emit(event, detail);
      else if (event) emit(event.type ?? event.event, event.payload ?? event.data ?? {});
    };

    async function importVoices() {
      if (!store.read().connected || openingPicker) return;
      openingPicker = true;
      $('import-voices').disabled = true;
      try {
        await safeCall('pick_and_import_voices');
      } finally {
        openingPicker = false;
        $('import-voices').disabled = false;
      }
    }

    for (const id of ['import-voices', 'empty-library', 'onboarding-import']) click(id, importVoices);
    for (const id of ['cable-guide', 'onboarding-install']) click(id, () => safeCall('open_url', CABLE_URL));
    click('assets-guide', () => safeCall('open_install_guide'));
    click('close-onboarding', async () => applySettings(await safeCall('dismiss_onboarding'), ['onboarding_dismissed']));
    click('settings-toggle', () => {
      if (!store.read().connected) return;
      settingsDialog.showModal();
      void refreshState();
    });
    click('settings-close', () => settingsDialog.close());
    settingsDialog.addEventListener('click', (event) => {
      if (event.target === settingsDialog) settingsDialog.close();
    });
    document.querySelector('[data-open-guide]').addEventListener('click', () => safeCall('open_install_guide'));
    click('open-logs', () => safeCall('open_logs'));
    click('power', () => {
      const state = store.read();
      if (state.engine.state === 'running') return withBusy('正在停止…', () => safeCall('stop'));
      if (!state.settings.selected_voice_id) return importVoices();
      return withBusy('正在加载音色…', () => safeCall('start'));
    });
    click('refresh-devices', () => withBusy('正在刷新设备…', async () => {
      if (await safeCall('list_devices')) {
        await refreshState();
        notify('音频设备已刷新');
      }
    }));

    for (const [key, id] of Object.entries(DEVICE_FIELDS)) {
      $(id).addEventListener('change', (event) => {
        const value = event.currentTarget.value;
        void withBusy('正在切换…', async () => {
          const selection = await safeCall('set_devices', { [key]: value === '' ? null : Number(value) });
          applySettings(selection, Object.keys(DEVICE_FIELDS));
        });
      });
    }
    $('monitor').addEventListener('change', async (event) => {
      const checked = event.currentTarget.checked;
      togglingMonitor = true;
      $('monitor').disabled = true;
      try {
        const settings = await safeCall('set_param', 'monitor', checked);
        if (settings && checked && !bridge.preview) notify('请佩戴耳机，以免产生回声。', 'warning');
        applySettings(settings, ['monitor']);
      } finally {
        togglingMonitor = false;
        $('monitor').disabled = false;
        $('monitor').checked = store.read().settings.monitor;
      }
    });
    $('gpu-index').addEventListener('change', (event) => {
      const value = Number(event.currentTarget.value);
      return withBusy('正在切换…', async () => applySettings(await safeCall('set_param', 'gpu_index', value), ['gpu_index']));
    });
    for (const button of document.querySelectorAll('[data-latency]')) {
      button.addEventListener('click', () => {
        const mode = button.dataset.latency;
        if (mode === store.read().settings.latency_mode) return;
        void withBusy('正在切换…', async () => applySettings(await safeCall('set_param', 'latency_mode', mode), ['latency_mode']));
      });
    }

    const updateSlider = {};
    for (const [parameter, [id, format]] of Object.entries(SLIDERS)) {
      const input = $(id);
      let latest;
      let syncing = false;
      let timer = null;
      async function flush() {
        clearTimeout(timer);
        timer = null;
        if (syncing) return;
        syncing = true;
        const requested = latest;
        const settings = await safeCall('set_param', parameter, requested);
        applySettings(settings, [parameter]);
        syncing = false;
        if (requested !== latest) {
          if (!timer) timer = setTimeout(flush, 70);
          return;
        }
        clearTimeout(timer);
        timer = null;
        pendingSliders.delete(parameter);
        setSliderDisplay(input, store.read().settings[parameter], format);
      }
      function update(value, immediate = false) {
        if (!store.read().connected || input.disabled) return;
        latest = value;
        pendingSliders.add(parameter);
        setSliderDisplay(input, latest, format);
        if (immediate) void flush();
        else if (!timer && !syncing) timer = setTimeout(flush, 70);
      }
      input.addEventListener('input', () => update(Number(input.value)));
      input.addEventListener('change', () => update(Number(input.value), true));
      input.addEventListener('dblclick', () => update(app.defaults[parameter], true));
      updateSlider[parameter] = update;
    }
    for (const button of document.querySelectorAll('[data-pitch]')) {
      button.addEventListener('click', () => updateSlider.pitch(Number(button.dataset.pitch), true));
    }

    const library = $('voice-library');
    const isFileDrop = (event) => Array.from(event.dataTransfer?.types ?? []).includes('Files');
    library.addEventListener('dragover', (event) => {
      if (!isFileDrop(event)) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = 'copy';
      library.dataset.dragging = 'true';
    });
    library.addEventListener('dragleave', (event) => {
      if (!library.contains(event.relatedTarget)) delete library.dataset.dragging;
    });
    library.addEventListener('drop', (event) => {
      if (!isFileDrop(event)) return;
      event.preventDefault();
      delete library.dataset.dragging;
      if (bridge?.preview) void safeCall('import_voice_paths', Array.from(event.dataTransfer.files, (file) => file.name));
    });
    document.addEventListener('dragover', (event) => { if (isFileDrop(event)) event.preventDefault(); });
    document.addEventListener('drop', (event) => { if (isFileDrop(event)) event.preventDefault(); });

    store.subscribe(render);
    render(store.read());
    try {
      bridge = await app.connect(emit);
      await refreshState();
      if (!store.read().connected) store.update({ engine: { state: 'error' } });
    } catch (error) {
      store.update({ engine: { state: 'error' } });
      notify(error.message || '无法连接桌面程序', 'error');
    }
  };
})();
