PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Middleware</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      background: #f3f4f6;
      color: #1c1e21;
    }
    main { max-width: 980px; margin: 0 auto; padding: 32px 20px 64px; }
    h1 { font-size: 28px; margin: 0 0 8px; }
    p { line-height: 1.5; }
    .lead { margin-top: 0; max-width: 68ch; }
    .status {
      display: inline-block;
      margin: 8px 0 20px;
      padding: 6px 10px;
      border-radius: 999px;
      background: #e7e9ee;
      font-size: 13px;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    section {
      background: white;
      border: 1px solid #e1e3e8;
      border-radius: 12px;
      padding: 18px;
    }
    label { display: block; font-size: 13px; font-weight: 650; margin: 12px 0 6px; }
    input, select, textarea {
      width: 100%;
      border: 1px solid #cfd3da;
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: white;
    }
    input[type="checkbox"] { width: auto; }
    .check { display: flex; align-items: center; gap: 8px; margin-top: 14px; }
    .check label { margin: 0; }
    button {
      margin-top: 16px;
      background: #1d2430;
      color: white;
      border: 0;
      border-radius: 8px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      background: #11151c;
      color: #e8edf5;
      border-radius: 8px;
      padding: 12px;
      min-height: 120px;
      font-family: Consolas, monospace;
      font-size: 13px;
    }
    img.preview { max-width: 100%; border-radius: 8px; margin-top: 12px; background: white; }
    .answer { min-height: 72px; }
    code { font-family: Consolas, monospace; font-size: 0.95em; }
    a { color: #1d2430; }
    @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <h1>LLM Middleware</h1>
    <p class="lead">One API for every application. Send model, type, query, and context. Use type claude, chatgpt, grok, groq, nvidia, ollama, or gemini. Groq and Grok are different providers. Ollama uses the online Ollama API. Try the calls in <a href="/swagger">Swagger</a>.</p>
    <div id="status" class="status">Checking middleware…</div>
    <div class="grid">
      <section>
        <form id="call">
          <label for="app">Application</label>
          <input id="app" name="app" value="application-one">

          <label for="type">Type</label>
          <select id="type" name="type">
            <option value="auto">auto</option>
            <option value="claude" selected>claude</option>
            <option value="chatgpt">chatgpt</option>
            <option value="grok">grok</option>
            <option value="nvidia">nvidia</option>
            <option value="groq">groq</option>
            <option value="ollama">ollama</option>
            <option value="gemini">gemini</option>
          </select>

          <label for="model">Model</label>
          <input id="model" name="model" value="auto">

          <label for="context">Context</label>
          <textarea id="context" name="context">You are a helpful assistant.</textarea>

          <label for="query">Query</label>
          <textarea id="query" name="query" required>Say hello. This call came from application one.</textarea>

          <div class="check">
            <input id="stream" name="stream" type="checkbox" checked>
            <label for="stream">Stream the answer</label>
          </div>

          <label for="apiKey">API key (only if you set MIDDLEWARE_API_KEY)</label>
          <input id="apiKey" name="apiKey" type="password" autocomplete="off">

          <button id="send" type="submit">Call API</button>
        </form>
      </section>
      <section>
        <label>Answer</label>
        <pre id="answer" class="answer">The answer will show here.</pre>
        <label>Full response</label>
        <pre id="out">{}</pre>
      </section>
    </div>
    <section style="margin-top: 20px">
      <h2>Generate an image</h2>
      <p>POST JSON to <code id="imageEndpoint">/api/v1/images</code>. Types are horde, gemini, ollama, chatgpt, grok, and nvidia.</p>
      <form id="imageCall">
        <label for="imageType">Type</label>
        <select id="imageType" name="imageType">
          <option value="auto">auto</option>
          <option value="nvidia" selected>nvidia</option>
          <option value="horde">horde</option>
          <option value="gemini">gemini</option>
          <option value="ollama">ollama</option>
          <option value="grok">grok</option>
          <option value="chatgpt">chatgpt</option>
        </select>
        <label for="imageModel">Model</label>
        <input id="imageModel" name="imageModel" value="auto">
        <label for="imagePrompt">Prompt</label>
        <textarea id="imagePrompt" name="imagePrompt" required>A red bicycle on a quiet street at sunset</textarea>
        <button id="imageSend" type="submit">Generate image</button>
      </form>
      <img id="imagePreview" class="preview" alt="" hidden>
      <pre id="imageOut">{}</pre>
    </section>
    <section style="margin-top: 20px">
      <h2>Call from another application</h2>
      <p>POST this JSON to <code id="endpoint">/api/v1/chat</code></p>
      <pre id="sample"></pre>
    </section>
  </main>
  <script>
    const endpoint = location.origin + "/api/v1/chat";
    document.getElementById("endpoint").textContent = endpoint;
    document.getElementById("imageEndpoint").textContent = location.origin + "/api/v1/images";

    function sample() {
      const body = {
        app: document.getElementById("app").value.trim() || "application-one",
        model: document.getElementById("model").value.trim() || "auto",
        type: document.getElementById("type").value,
        query: document.getElementById("query").value,
        context: document.getElementById("context").value,
        stream: document.getElementById("stream").checked
      };
      return body;
    }

    function refreshSample() {
      document.getElementById("sample").textContent = JSON.stringify(sample(), null, 2);
    }

    ["app", "type", "model", "context", "query", "stream"].forEach((id) => {
      document.getElementById(id).addEventListener("input", refreshSample);
      document.getElementById(id).addEventListener("change", refreshSample);
    });
    refreshSample();

    async function readStream(res) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      document.getElementById("answer").textContent = "";
      document.getElementById("out").textContent = "Streaming…";
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        const parts = buffer.split("\\n\\n");
        buffer = parts.pop() || "";
        for (const part of parts) {
          const dataLine = part.split("\\n").filter((line) => line.startsWith("data:")).map((line) => line.slice(5).trim()).join("");
          if (!dataLine || dataLine === "[DONE]") continue;
          const eventLine = part.split("\\n").find((line) => line.startsWith("event:"));
          const eventName = eventLine ? eventLine.slice(6).trim() : "";
          const payload = JSON.parse(dataLine);
          if (eventName === "delta") {
            answer += payload.text || "";
            document.getElementById("answer").textContent = answer;
          } else if (eventName === "done" || eventName === "start") {
            document.getElementById("out").textContent = JSON.stringify(payload, null, 2);
          }
        }
      }
      if (!answer) document.getElementById("answer").textContent = "No answer";
    }

    fetch("/api/v1/providers")
      .then((res) => res.json())
      .then((data) => {
        const mode = data.mode === "mock" ? "Mock mode (no real model call)" : "Live mode";
        const auth = data.auth_required ? "API key required" : "API key not required on this computer";
        document.getElementById("status").textContent = mode + " · " + auth;
      })
      .catch(() => {
        document.getElementById("status").textContent = "Middleware is not reachable";
      });

    document.getElementById("call").addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = document.getElementById("send");
      button.disabled = true;
      document.getElementById("answer").textContent = "Calling…";
      const headers = { "Content-Type": "application/json" };
      const key = document.getElementById("apiKey").value.trim();
      if (key) headers["X-API-Key"] = key;
      try {
        const res = await fetch("/api/v1/chat", {
          method: "POST",
          headers,
          body: JSON.stringify(sample())
        });
        if (document.getElementById("stream").checked && res.ok && (res.headers.get("content-type") || "").includes("text/event-stream")) {
          await readStream(res);
          return;
        }
        const data = await res.json();
        document.getElementById("out").textContent = JSON.stringify(data, null, 2);
        document.getElementById("answer").textContent =
          data.answer || (data.error && data.error.message) || data.detail || "No answer";
      } catch (error) {
        document.getElementById("answer").textContent = "Call failed";
        document.getElementById("out").textContent = String(error);
      } finally {
        button.disabled = false;
      }
    });

    document.getElementById("imageCall").addEventListener("submit", async (event) => {
      event.preventDefault();
      const button = document.getElementById("imageSend");
      const preview = document.getElementById("imagePreview");
      button.disabled = true;
      preview.hidden = true;
      document.getElementById("imageOut").textContent = "Generating…";
      const headers = { "Content-Type": "application/json" };
      const key = document.getElementById("apiKey").value.trim();
      if (key) headers["X-API-Key"] = key;
      try {
        const res = await fetch("/api/v1/images", {
          method: "POST",
          headers,
          body: JSON.stringify({
            app: "image-app",
            model: document.getElementById("imageModel").value.trim() || "auto",
            type: document.getElementById("imageType").value,
            prompt: document.getElementById("imagePrompt").value,
            size: document.getElementById("imageType").value === "horde" ? "512x512" : "1024x1024"
          })
        });
        const data = await res.json();
        const picture = data.images && data.images[0];
        const shown = Object.assign({}, data);
        if (shown.images) {
          shown.images = shown.images.map((image) => ({
            url: image.url || null,
            b64_json: image.b64_json ? image.b64_json.slice(0, 24) + "…" : null
          }));
        }
        document.getElementById("imageOut").textContent = JSON.stringify(shown, null, 2);
        if (picture && picture.url) {
          preview.src = picture.url;
          preview.hidden = false;
        } else if (picture && picture.b64_json) {
          preview.src = "data:image/png;base64," + picture.b64_json;
          preview.hidden = false;
        }
      } catch (error) {
        document.getElementById("imageOut").textContent = String(error);
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""
