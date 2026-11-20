(() => {
  'use strict';

  const app = window.MaskVox;
  app.voiceSpec = (voice) => `${voice.version} · ${voice.sr / 1000}k`;
  function avatarGradient(name) {
    let hash = 0;
    for (const char of name) hash = Math.imul(37, hash) + char.codePointAt(0) | 0;
    const hue = (hash >>> 0) % 360;
    return `linear-gradient(140deg, hsl(${hue} 43% 44%), hsl(${(hue + 41) % 360} 42% 29%))`;
  }

  app.createVoiceLibrary = (root, templates, actions) => {
    const cards = new Map();
    let openMenu = null;
    let lastOrdering = '';
    function closeMenu(focus = false) {
      if (!openMenu) return;
      openMenu.menu.hidden = true;
      openMenu.more.setAttribute('aria-expanded', 'false');
      if (focus) openMenu.more.focus();
      openMenu = null;
    }
    document.addEventListener('click', (event) => {
      if (openMenu && !openMenu.root.contains(event.target)) closeMenu();
    });
    document.addEventListener('keydown', (event) => {
      if (event.key === 'Escape' && openMenu) closeMenu(true);
    });
    function makeCard(voice) {
      const cardRoot = templates.voice.content.firstElementChild.cloneNode(true);
      const card = {
        root: cardRoot, voice,
        choose: cardRoot.querySelector('.voice-select'),
        more: cardRoot.querySelector('.voice-more'),
        menu: cardRoot.querySelector('.voice-menu'),
        avatar: cardRoot.querySelector('.voice-avatar'),
        name: cardRoot.querySelector('.voice-name'),
        meta: cardRoot.querySelector('.voice-meta'),
      };
      card.choose.addEventListener('click', () => {
        closeMenu();
        actions.select(card.voice);
      });
      card.more.addEventListener('click', () => {
        if (openMenu === card) return closeMenu();
        closeMenu();
        openMenu = card;
        card.menu.hidden = false;
        card.more.setAttribute('aria-expanded', 'true');
        const menuRect = card.menu.getBoundingClientRect();
        const viewRect = root.parentElement.getBoundingClientRect();
        card.menu.dataset.up = String(menuRect.bottom > viewRect.bottom && cardRoot.getBoundingClientRect().top - menuRect.height > viewRect.top);
      });
      card.menu.addEventListener('click', (event) => {
        const button = event.target.closest('button[data-action]');
        if (!button) return;
        closeMenu(true);
        actions[button.dataset.action](card.voice);
      });
      return card;
    }

    return (voices, selected, loading, newIds) => {
      const alive = new Set(voices.map((voice) => voice.id));
      for (const id of cards.keys()) {
        if (alive.has(id)) continue;
        if (openMenu === cards.get(id)) closeMenu();
        cards.delete(id);
      }
      const rows = voices.map((voice) => {
        if (!cards.has(voice.id)) cards.set(voice.id, makeCard(voice));
        const card = cards.get(voice.id);
        card.voice = voice;
        card.root.dataset.selected = String(voice.id === selected);
        card.choose.setAttribute('aria-pressed', String(voice.id === selected));
        card.choose.disabled = loading;
        if (newIds.delete(voice.id)) {
          card.root.dataset.new = 'true';
          card.root.addEventListener('animationend', () => delete card.root.dataset.new, { once: true });
        }
        if (card.name.textContent !== voice.name) {
          card.name.textContent = voice.name;
          card.name.title = voice.name;
          card.more.setAttribute('aria-label', `${voice.name}的音色操作`);
          card.avatar.textContent = Array.from(voice.name.trim())[0] ?? '音';
          card.avatar.style.background = avatarGradient(voice.name);
        }
        const metaText = `${app.voiceSpec(voice)} · ${voice.has_index ? '有索引' : '无索引'}`;
        if (card.meta.textContent !== metaText) {
          const label = document.createElement('span');
          label.className = voice.has_index ? 'voice-meta-index' : '';
          label.textContent = voice.has_index ? '有索引' : '无索引';
          card.meta.replaceChildren(`${app.voiceSpec(voice)} · `, label);
        }
        return card.root;
      });
      const ordering = JSON.stringify(voices.map((voice) => voice.id));
      if (ordering !== lastOrdering) root.replaceChildren(...rows);
      lastOrdering = ordering;
    };
  };
})();
