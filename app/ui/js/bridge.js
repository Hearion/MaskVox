(() => {
  'use strict';

  const app = window.MaskVox;
  const STORAGE_KEY = 'maskvox-preview-v1';
  const sampleVoice = (id, name, version, sr, has_index) => ({
    id, name, version, sr, f0: 1, has_index, vec_dim: version === 'v2' ? 768 : 256,
  });

  function createMock(emit) {
    const initialVoices = [
      sampleVoice('mock-01', '暮青', 'v2', 40000, true),
      sampleVoice('mock-02', '雾岛', 'v2', 48000, false),
      sampleVoice('mock-03', '砂糖', 'v1', 40000, true),
    ];
    const devices = {
      inputs: [{ id: 3, name: '麦克风（USB 音频设备）' }, { id: 7, name: '阵列麦克风' }],
      outputs: [{ id: 4, name: '扬声器（高清音频）' }, { id: 5, name: '耳机（USB 音频设备）' }],
      default_input_id: 3, default_output_id: 4, cable_output_id: null, has_vb_cable: false,
    };
    let saved;
    try {
      saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    } catch {
      saved = null;
    }
    const settings = {
      ...app.defaults, input_id: 3, output_id: 4, monitor_id: 5,
      selected_voice_id: initialVoices[0].id, ...saved?.settings,
    };
    let voices = saved?.voices ?? initialVoices;
    let engine = { state: 'idle', message: '', backend: null };
    let meterTimer = null;
    let loadingTimer = null;
    const imports = new Map();
    const delay = (callback, time) => window.setTimeout(callback, time);
    const equal = (item, id) => item.id === id;
    const save = () => {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ settings, voices }));
      } catch {
        // Private browsing can disable local-file storage.
      }
    };
    const status = (state, message = '') => {
      engine = { state, message, backend: state === 'idle' ? null : (settings.gpu_index === -1 ? 'CPU' : 'DirectML') };
      emit('status', engine);
    };
    function stop() {
      clearTimeout(loadingTimer);
      clearInterval(meterTimer);
      status('idle');
    }
    function start(message = '正在加载音色…') {
      if (!settings.selected_voice_id) throw new Error('请先导入并选择一个音色');
      clearTimeout(loadingTimer);
      clearInterval(meterTimer);
      status('loading', message);
      loadingTimer = delay(() => {
        status('running');
        let tick = 0;
        meterTimer = setInterval(() => {
          const mix = Math.sin(++tick * .49) * .42 + Math.sin(tick * 1.51) * .19 + .42;
          const infer_ms = 35 + Math.round(4 * Math.sin(tick * .19));
          const block = { low: 120, balanced: 200, high: 350 }[settings.latency_mode];
          emit('meter', { in_db: -58 + 42 * mix, out_db: -63 + 46 * mix, infer_ms,
            latency_ms: block + infer_ms + 24, underruns: 0 });
        }, 100);
      }, 470);
    }
    function reconfigure() {
      if (engine.state === 'running' || engine.state === 'loading') start('正在切换…');
    }
    function importPaths(paths) {
      const file = paths?.find((path) => /\.(pth|zip|maskvox)$/i.test(path));
      if (!file) throw new Error('请选择 .pth、.zip 或 .maskvox 音色文件');
      const name = file.split(/[\\/]/).at(-1);
      const job_id = `preview-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      const publish = (stage, progress, fields = {}) => emit('import_progress', { job_id, file: name, stage, progress, ...fields });
      const timers = [delay(() => publish('loading', .08), 10), delay(() => publish('exporting', .48), 430),
        delay(() => publish('verifying', .88), 970), delay(() => {
          const voice = sampleVoice(job_id, name.replace(/\.[^.]+$/, ''), 'v2', 40000, true);
          voices = [voice, ...voices];
          settings.selected_voice_id = voice.id;
          save();
          publish('done', 1, { voice_id: voice.id });
          emit('voices_changed', {});
          imports.delete(job_id);
          reconfigure();
        }, 1370)];
      imports.set(job_id, { timers, publish });
      return { job_id };
    }
    const mock = {
      get_state: () => ({
        settings, voices, devices, engine, has_vb_cable: false, base_models_ready: true,
        converter_installed: true, logs_path: 'logs/maskvox.log', version: '1.0.0',
        gpu_adapters: [{ id: 0, name: 'NVIDIA GeForce RTX 3070' }],
      }),
      list_devices: () => devices,
      list_voices: () => voices,
      start: () => start(), stop,
      set_devices(values) { Object.assign(settings, values); save(); reconfigure(); return settings; },
      set_param(name, value) {
        settings[name] = value;
        save();
        if (name === 'latency_mode' || name === 'gpu_index') reconfigure();
        if (name === 'monitor' && value) emit('warning', { code: 'headphones', message: '请佩戴耳机，以免产生回声。' });
        return settings;
      },
      select_voice(id) { settings.selected_voice_id = id; save(); reconfigure(); },
      rename_voice(id, name) { voices = voices.map((voice) => equal(voice, id) ? { ...voice, name } : voice); save(); },
      delete_voice(id) {
        voices = voices.filter((voice) => !equal(voice, id));
        if (settings.selected_voice_id === id) {
          stop();
          settings.selected_voice_id = voices[0]?.id ?? null;
        }
        save();
      },
      export_voice: (id) => `${voices.find((voice) => equal(voice, id))?.name ?? '音色'}.maskvox`,
      pick_and_import_voices: () => importPaths([`试听音色 ${voices.length + 1}.pth`]),
      import_voice_paths: importPaths,
      cancel_import(id) {
        const job = imports.get(id);
        if (!job) return;
        job.timers.forEach(clearTimeout);
        job.publish('cancelled', 0);
        imports.delete(id);
      },
      open_url(url) { window.open(url, '_blank', 'noopener,noreferrer'); },
      open_logs() { emit('warning', { message: '启动桌面程序后可以打开实际日志目录。' }); },
      open_install_guide() { emit('warning', { message: '在项目目录的 README.md 中查看安装方法。' }); },
      dismiss_onboarding() { settings.onboarding_dismissed = true; save(); return settings; },
    };
    return { preview: true, call: async (method, ...args) => structuredClone(mock[method](...args)) };
  }

  async function nativeApi() {
    if (window.pywebview?.api) return window.pywebview.api;
    if (!window.chrome?.webview) return null;
    await new Promise((resolve) => {
      const timeout = setTimeout(resolve, 10000);
      window.addEventListener('pywebviewready', () => {
        clearTimeout(timeout);
        resolve();
      }, { once: true });
    });
    if (!window.pywebview?.api) throw new Error('未能连接到桌面程序，请重新启动 MaskVox');
    return window.pywebview.api;
  }

  app.connect = async (emit) => {
    const api = await nativeApi();
    if (!api) return createMock(emit);
    return {
      preview: false,
      async call(method, ...args) {
        const response = await api[method](...args);
        if (response?.ok) return response.data;
        const error = new Error(response?.error ?? '请求失败，请查看日志');
        error.code = response?.code;
        throw error;
      },
    };
  };
})();
