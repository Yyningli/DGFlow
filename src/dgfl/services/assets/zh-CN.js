/* DGFlow's local UI adapter for TensorBoard 2.21. Native data and state stay native. */
(function (root) {
  'use strict';
  const words = Object.freeze({
    'Time Series': '时间序列', 'TIME SERIES': '时间序列', 'Scalars': '标量',
    'SCALARS': '标量', 'Scalar': '标量', 'scalars': '标量', 'Image': '图像', 'Images': '图像',
    'Inactive': '未启用面板', 'INACTIVE': '未启用面板', 'inactive': '未启用面板',
    'HParams': '超参数', 'HPARAMS': '超参数', 'hparams': '超参数',
    'Histogram': '直方图', 'Histograms': '直方图', 'All': '全部',
    'Runs': '实验记录', 'Run': '实验记录', 'Run Name': '实验名称',
    'Run name': '实验名称', 'Name': '名称', 'Color': '颜色', 'Selected': '已选择',
    'Experiment': '实验', 'Experiments': '实验', 'Experiment Name': '实验名称',
    'Filter': '筛选', 'Filter runs (regex)': '筛选实验（正则表达式）',
    'Filter tags (regex)': '筛选指标（正则表达式）', 'Filter tags': '筛选指标',
    'Filter by tag': '按指标筛选', 'Search': '搜索', 'Settings': '设置',
    'General': '通用', 'Horizontal Axis': '横轴', 'Step': '步数',
    'Value': '数值', 'Smoothed': '平滑值', 'Smoothing': '平滑系数',
    'Relative': '相对时间', 'Relative time': '相对时间', 'Wall': '实际时间',
    'Wall time': '实际时间', 'Wall Time': '实际时间', 'Time': '时间',
    'Card Width': '图表宽度', 'Tooltip sorting method': '提示框排序方式',
    'Default': '默认', 'Ascending': '升序', 'Descending': '降序',
    'Alphabetical': '按名称', 'Nearest': '最近值', 'Nearest Pixel': '最近像素',
    'Nearest Y': '最近纵轴值',
    'Offset': '偏移', 'Overlay': '叠加',
    'Enable step selection and data table': '启用步数选择和数据表',
    '(Scalars only)': '（仅标量）', 'Enable Range Selection': '启用范围选择',
    'Enable saving pins (Scalars only)': '保存固定图表（仅标量）',
    'Ignore outliers in chart scaling': '缩放图表时忽略离群值',
    'Limit tooltip rows to': '提示框最多显示行数',
    'Partition non-monotonic X axis': '分段显示非单调横轴',
    'Brightness': '亮度', 'Contrast': '对比度', 'Reset card width': '重置图表宽度',
    'Reset brightness': '重置亮度', 'Reset contrast': '重置对比度',
    'Pinned': '已固定', 'Pin card': '固定图表', 'Unpin card': '取消固定图表',
    'Clear all pins': '取消全部固定', 'New card pinned': '已固定新图表',
    'Pin cards for a quick view and comparison': '固定图表以便快速查看和比较',
    'Disable Saving Pins?': '关闭图表固定保存？', 'Disable': '关闭',
    'Disabling saving pins will remove locally stored pinned card data and no longer allow pinned cards to be shared across multiple experiments.':
      '关闭后将移除本地保存的固定图表，无法在多个实验之间共享固定图表。',
    'You can re-enable the feature and re-pin cards at any time.': '可随时重新启用并固定图表。',
    'Fit line chart domains to data': '根据数据调整图表范围',
    'Toggle full size mode': '切换全尺寸模式', 'More line chart options': '更多图表选项',
    'Toggle Y-axis log scale': '切换纵轴对数刻度',
    'Toggle Y-axis log scale on line chart': '切换图表纵轴对数刻度',
    'Download data': '下载数据', 'Open dialog to download data': '打开数据下载窗口',
    'Download as…': '下载为…', 'Download as...': '下载为…',
    'Select a run to download a data for a series': '选择要下载数据的实验记录',
    'Edit Table Columns': '编辑表格列',
    'Open menu to edit data table columns': '编辑数据表列',
    'Expand Table': '展开数据表', 'Collapse Table': '收起数据表',
    'Expand group': '展开指标组', 'Collapse group': '收起指标组',
    'Deselect fob': '取消步数选择', 'Edit step': '编辑步数',
    'Toggle settings side pane': '打开或关闭设置面板',
    'Close side pane': '关闭设置面板',
    'Menu for changing light or dark theme': '切换浅色或深色主题',
    'Browser default': '跟随浏览器', 'Light': '浅色', 'Dark': '深色',
    'Set the theme to match the default mode in the browser.': '使用浏览器默认主题。',
    'Force light TensorBoard theme.': '使用 TensorBoard 浅色主题。',
    'Force dark TensorBoard theme.': '使用 TensorBoard 深色主题。',
    'Current mode: [Browser default]. Switch between browser default, light, or dark theme.': '当前：跟随浏览器。可切换浅色、深色或跟随浏览器主题。',
    'Current mode: [Light mode]. Switch between browser default, light, or dark theme.': '当前：浅色。可切换浅色、深色或跟随浏览器主题。',
    'Current mode: [Dark mode]. Switch between browser default, light, or dark theme.': '当前：深色。可切换浅色、深色或跟随浏览器主题。',
    'Help': '帮助',
    'Alt + Scroll to Zoom': '按住 Alt 并滚动以缩放',
    'Color runs by...': '实验颜色分组…', 'Color runs by': '实验颜色分组',
    'Color runs by regex': '按正则表达式设置实验颜色', 'Color group preview': '颜色分组预览',
    'Regex': '正则表达式', 'Regex type': '正则表达式类型', '(none set)': '（未设置）',
    'Enter a regex with capturing groups to match against run names:':
      '输入带捕获组的正则表达式以匹配实验名称：',
    'By Experiment Name': '按实验名称', 'By Run Name': '按记录名称',
    'Include Undefined': '包含未定义值', 'Insert Column Left': '在左侧插入列',
    'Insert Column Right': '在右侧插入列', 'Sort Ascending': '升序排列',
    'Sort Descending': '降序排列', 'Remove': '移除', 'Mode': '模式',
    'Next': '下一页', 'Previous': '上一页', 'Close': '关闭', 'Cancel': '取消',
    'Save': '保存', 'Change': '更改', 'Apply': '应用', 'Refresh': '刷新',
    'Reload': '刷新', 'Reload data': '刷新数据', 'Refresh data': '刷新数据',
    'Reload Period (seconds)': '刷新间隔（秒）', 'Pagination Limit': '每页图表数量',
    'Page size has to be a positive integer.': '每页数量必须为正整数。',
    'Only available when Horizontal Axis is set to step': '仅在横轴设为步数时可用',
    'Tooltip rows limit': '提示框行数上限',
    'When saving pins are enabled, pinned cards will be visible across multiple experiments.':
      '启用保存后，固定图表可在多个实验之间查看。',
    'Enable auto-reload': '启用自动刷新', 'Enable auto reload': '启用自动刷新',
    'Reload interval': '刷新间隔', 'Reload period': '刷新间隔', 'seconds': '秒',
    'Auto-reload': '自动刷新', 'Auto reload': '自动刷新',
    'Automatic data reload': '自动刷新数据',
    'Loading...': '加载中…', 'Loading…': '加载中…',
    'Data failed to load.': '数据加载失败。', 'Data could not be loaded.': '无法加载数据。',
    'No Actions Available': '没有可用操作', 'No Matching Values': '没有匹配值',
    'No matches for tag filter': '没有匹配的指标',
    'There are no runs matching the regex': '没有匹配该正则表达式的实验记录',
    'No runs are in the group': '此分组中没有实验记录',
    'Tags matching filter': '匹配筛选条件的指标',
    'No dashboards are active for the current data set.': '当前数据集没有可用的面板。',
    'You can select a dashboard from the list above.': '可从上方列表选择面板。',
    'Show actual image size': '显示图像原始大小', 'Log directory:': '日志目录：',
    'Probable causes:': '可能的原因：', 'Load all anyway': '仍然加载全部',
    'Warning: There were too many columns to load all of them efficiently.': '列数过多，无法高效加载全部列。',
    'TensorBoard can’t find your event files.': 'TensorBoard 未找到事件文件。',
    'TensorBoard tutorial': 'TensorBoard 使用指南',
    'The TensorBoard server may be down or inaccessible.': 'TensorBoard 服务可能已停止或无法访问。',
    'You haven’t written any data to your event files.': '事件文件中尚未写入数据。'
  });

  // Never translate experiment names, user tags, regexes, chart values, or input
  // values even when they happen to equal an English UI label such as "Run".
  const protectedSelector = [
    'script', 'style', 'code', 'pre', 'textarea', 'svg', 'canvas',
    '[contenteditable="true"]', 'card-run-name', 'card-run-name-component',
    'tb-experiment-alias', 'tb-truncated-path', '.tag', '.name', '.run-name',
    '.display-regex-string', '.legend', 'td', 'tb-data-table-content-row',
    'tb-data-table-content-cell', 'filterbar', 'filterbar-component',
    'mat-option', '[role="option"]'
  ].join(',');
  const uiSelector = [
    'settings-dialog-component', 'settings-dialog', 'settings-polymer-interop',
    'settings-button-component', 'settings-button', 'plugin-selector-component',
    'metrics-dashboard-settings-component', 'metrics-dashboard-settings',
    'metrics-dashboard-right-pane', 'scalar-card-component', 'scalar-card',
    'scalar-card-data-table', 'metrics-scalar-column-editor-component',
    'metrics-card-group-toolbar-component', 'metrics-card-group-toolbar',
    'metrics-empty-tag-match-component', 'metrics-tag-filter-component',
    'metrics-pinned-view-component', 'metrics-main-view-component',
    'runs-data-table', 'runs-table', 'runs-selector-component', 'runs-selector',
    'runs-group-menu-button-component', 'data_download_dialog_component',
    'data_download_dialog', 'app-header', 'app-header-reload', 'tb-filter-input', 'tb-dropdown',
    'tb-data-table-filter', 'button', '[role="button"]', '[role="tab"]',
    '[role="menuitem"]', '[role="menuitemradio"]', '[role="tooltip"]',
    'mat-label', 'th', 'tb-data-table-header-cell', '[role="dialog"]'
  ].join(',');
  const attributes = ['aria-label', 'title', 'placeholder', 'alt'];

  function translate(value) {
    if (typeof value !== 'string') return value;
    const key = value.trim();
    if (!Object.prototype.hasOwnProperty.call(words, key)) {
      // These native UI templates interpolate one numeric reading. Only use
      // these anchored patterns after the same UI/data-ancestor checks.
      if (/^\d[\d,.]* cards$/.test(key)) return value.replace('cards', '个图表');
      if (/^Link by step(?:\s+-?\d+(?:\.\d+)?)?$/.test(key)) return value.replace('Link by step', '按步数关联');
      if (/^Reload period has to be minimum of \d+ seconds\.$/.test(key)) {
        return value.replace(/Reload period has to be minimum of (\d+) seconds\./, '刷新间隔至少为 $1 秒。');
      }
      if (/^Last Updated: .+$/.test(key)) return value.replace('Last Updated:', '最近更新：');
      return value;
    }
    const start = value.indexOf(key);
    return value.slice(0, start) + words[key] + value.slice(start + key.length);
  }
  function knownSettingsOption(element) {
    if (!element || !element.ownerDocument) return false;
    const option = element.closest('mat-option,[role="option"]');
    const panel = option && option.closest('[role="listbox"]');
    if (!panel || !panel.id) return false;
    // Angular Material renders options outside their trigger component. Link
    // the panel back to a fixed native settings control, never by option text.
    const controls = element.ownerDocument.querySelectorAll(
      'metrics-dashboard-settings-component .x-axis-type mat-select,' +
      'metrics-dashboard-settings-component .tooltip-sort mat-select,' +
      'metrics-dashboard-settings-component .histogram-mode mat-select,' +
      'plugin-selector-component mat-select');
    return Array.from(controls).some(function (control) {
      return control.getAttribute('aria-controls') === panel.id;
    });
  }
  function eligible(element) {
    return Boolean(element && (knownSettingsOption(element) ||
      (!element.closest(protectedSelector) && element.closest(uiSelector))));
  }
  function metricLabel(tag, catalog) {
    if (typeof tag !== 'string' || !catalog || !catalog.fields || !catalog.groups) return null;
    const parts = tag.split('/');
    if (parts.length < 2 || parts.length > 8) return null;
    let field = parts.pop();
    const maximum = field.endsWith('_max');
    if (maximum) field = field.slice(0, -4);
    if (!Object.prototype.hasOwnProperty.call(catalog.fields, field)) return null;
    const labels = [];
    for (const part of parts) {
      if (Object.prototype.hasOwnProperty.call(catalog.groups, part)) labels.push(catalog.groups[part]);
      else if (/^(?:client|authority|aggregator)[1-9][0-9]*$/.test(part) ||
               (Array.isArray(catalog.kernels) && catalog.kernels.includes(part))) labels.push(part);
      else return null;
    }
    const entry = catalog.fields[field];
    const last = entry.label + (maximum ? '（各边缘进程最大值）' : '') + '（' + entry.unit + '）';
    return {prefix: labels.join(' / '), suffix: ' / ' + last};
  }
  function groupLabel(path, catalog) {
    if (typeof path !== 'string' || !catalog || !catalog.groups) return null;
    const parts = path.split('/');
    if (!parts.length || parts.length > 7 ||
        parts.some(function (part) { return !Object.prototype.hasOwnProperty.call(catalog.groups, part); })) return null;
    return parts.map(function (part) { return catalog.groups[part]; }).join(' / ');
  }

  function evaluationLegend(group, document) {
    if (group.getAttribute('title') !== 'evaluation') return;
    const container = group.closest('.card-group');
    const toolbar = group.closest('metrics-card-group-toolbar');
    if (!container || !toolbar || toolbar.parentElement !== container ||
        container.querySelector(':scope > .dgflow-evaluation-legend')) return;
    const note = document.createElement('p');
    note.className = 'dgflow-evaluation-legend';
    note.textContent = '曲线说明：实验根记录 = 聚合模型·全局测试集；' +
      'nodes/clientN（有记录时）= 客户端·本地训练集，每轮最后一个 epoch。' +
      '横轴均为全局轮次，颜色对应左侧实验记录。两者使用不同评估集。';
    container.insertBefore(note, toolbar.nextSibling);
  }

  function start(document, window) {
    if (window.__dgflowTensorBoardChinese) return window.__dgflowTensorBoardChinese;
    let observer;
    let pending = new Set();
    let frame = null;
    let active = false;
    let catalog = null;
    const options = {subtree: true, childList: true, characterData: true,
                     attributes: true, attributeFilter: attributes};
    document.documentElement.lang = 'zh-CN';
    function metricTitles(element) {
      if (!catalog) return;
      const selector = 'scalar-card-component tb-truncated-path.tag,scalar-card tb-truncated-path.tag';
      const titles = element.matches && element.matches(selector) ? [element] : [];
      if (element.querySelectorAll) titles.push(...element.querySelectorAll(selector));
      for (const title of titles) {
        const label = metricLabel(title.getAttribute('title'), catalog);
        if (!label) continue;
        const first = title.querySelector('.first-text-part');
        const second = title.querySelector('.second-text-part');
        // Preserve Angular's component/spans and its underlying raw title/value.
        if (first && second) {
          if (first.textContent !== label.prefix) first.textContent = label.prefix;
          if (second.textContent !== label.suffix) second.textContent = label.suffix;
        } else if (second && second.textContent !== label.prefix + label.suffix) {
          second.textContent = label.prefix + label.suffix;
        }
        const accessible = label.prefix + label.suffix;
        if (title.getAttribute('aria-label') !== accessible) title.setAttribute('aria-label', accessible);
      }
      const groups = element.matches && element.matches('metrics-card-group-toolbar-component .group-title') ? [element] : [];
      if (element.querySelectorAll) groups.push(...element.querySelectorAll('metrics-card-group-toolbar-component .group-title'));
      for (const group of groups) {
        const label = groupLabel(group.getAttribute('title'), catalog);
        if (label && group.textContent !== label) group.textContent = label;
        evaluationLegend(group, document);
      }
    }
    function elementAttributes(element) {
      if (!eligible(element)) return;
      for (const name of attributes) {
        const original = element.getAttribute(name);
        if (original === null) continue;
        const translated = translate(original);
        if (translated !== original) element.setAttribute(name, translated);
      }
      if (element.matches('settings-button-component button,settings-button button') &&
          !element.hasAttribute('aria-label')) element.setAttribute('aria-label', 'TensorBoard 设置');
      if (element.matches('tb-dashboard-layout .full-screen-btn') &&
          !element.hasAttribute('aria-label')) element.setAttribute('aria-label', '切换实验列表显示');
      if (element.matches('tb-dashboard-layout .expand-collapsed-sidebar') &&
          !element.hasAttribute('aria-label')) element.setAttribute('aria-label', '展开实验列表');
    }
    function adapt(element) {
      if (!element || element.nodeType !== 1) return;
      elementAttributes(element);
      const walker = document.createTreeWalker(element, 1 | 4);
      let node;
      while ((node = walker.nextNode())) {
        if (node.nodeType === 1) elementAttributes(node);
        else if (eligible(node.parentElement)) {
          const translated = translate(node.nodeValue);
          if (translated !== node.nodeValue) node.nodeValue = translated;
        }
      }
      metricTitles(element);
    }
    function flush() {
      frame = null;
      if (!active) return;
      observer.disconnect();
      const roots = pending;
      pending = new Set();
      try {
        for (const element of roots) if (element.isConnected) adapt(element);
      } finally {
        if (active) observer.observe(document.documentElement, options);
      }
    }
    function queue(element) {
      if (!active || !element) return;
      if (pending.size > 128) pending = new Set([document.documentElement]);
      else pending.add(element.nodeType === 1 ? element : element.parentElement);
      if (frame === null) frame = window.requestAnimationFrame(flush);
    }
    observer = new window.MutationObserver(function (records) {
      for (const record of records) {
        // Changes within a protected tag still need the dedicated title adapter.
        const parent = record.target.nodeType === 1 ? record.target : record.target.parentElement;
        queue(parent && parent.closest('tb-truncated-path.tag') || parent);
      }
    });
    function resume() {
      if (active) return;
      active = true;
      observer.observe(document.documentElement, options);
      queue(document.documentElement);
    }
    function close() {
      active = false;
      observer.disconnect();
      if (frame !== null) window.cancelAnimationFrame(frame);
      frame = null;
      pending.clear();
    }
    window.addEventListener('pagehide', close);
    window.addEventListener('pageshow', resume);
    const controller = {close: close, resume: resume};
    window.__dgflowTensorBoardChinese = controller;
    resume();
    // A failure affects titles only; controls stay translated and business/TB
    // requests do not depend on this optional, bounded catalog request.
    if (typeof window.fetch === 'function') {
      window.fetch('/tensorboard/dgflow-i18n/labels.json', {credentials: 'same-origin'})
        .then(function (response) { if (!response.ok) throw new Error('catalog'); return response.json(); })
        .then(function (value) { catalog = value; queue(document.documentElement); })
        .catch(function () {});
    }
    return controller;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = {translate: translate, eligible: eligible, metricLabel: metricLabel,
                      groupLabel: groupLabel, evaluationLegend: evaluationLegend, start: start};
  }
  if (root && root.document) {
    if (root.document.readyState === 'loading') {
      root.document.addEventListener('DOMContentLoaded', function () { start(root.document, root); }, {once: true});
    } else start(root.document, root);
  }
})(typeof globalThis !== 'undefined' ? globalThis.window : null);
