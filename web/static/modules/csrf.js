/* Double-submit CSRF token injection for cookie-authenticated mutations.
   The backend requires X-CSRF-Token to match the shorts_csrf cookie on
   cookie-authenticated POST/PUT/PATCH/DELETE requests; this module installs
   a fetch wrapper so every in-app mutation carries the header. Bearer-token
   clients and the SSE/polling GET paths are unaffected. */
(() => {
  const MUTATING = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);
  const previousFetch = window.fetch.bind(window);

  function csrfToken() {
    const entry = String(document.cookie || '')
      .split(/;\s*/)
      .find(item => item.startsWith('shorts_csrf='));
    return entry ? decodeURIComponent(entry.slice('shorts_csrf='.length)) : '';
  }

  window.fetch = (input, init = {}) => {
    try {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const method = String((init && init.method) || (input && input.method) || 'GET').toUpperCase();
      const target = new URL(url, window.location.href);
      const token = csrfToken();
      if (MUTATING.has(method) && token && target.origin === window.location.origin) {
        const headers = new Headers((init && init.headers) || (input && input.headers) || {});
        if (!headers.has('X-CSRF-Token')) headers.set('X-CSRF-Token', token);
        init = Object.assign({}, init, { headers });
      }
    } catch (_) {
      /* Header injection is best-effort; the server still answers 403 with a
         retryable code if the token is missing. */
    }
    return previousFetch(input, init);
  };
})();
