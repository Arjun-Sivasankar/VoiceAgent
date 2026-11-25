const url = "ws://127.0.0.1:8765/v1/realtime";
const ws = new WebSocket(url);
const seen = new Map();
const t0 = Date.now();
const log = (message) => console.log(((Date.now() - t0) / 1000).toFixed(1) + "s " + message);

ws.onopen = () => {
  log("open");
  ws.send(JSON.stringify({
    type: "session.update",
    session: {
      type: "realtime",
      instructions: "You are a helpful voice assistant. Reply in one short sentence.",
      audio: {
        input: { turn_detection: { type: "server_vad", interrupt_response: true } },
        output: {}
      }
    }
  }));
};

ws.onmessage = (event) => {
  let ev;
  try { ev = JSON.parse(event.data); } catch { return; }
  seen.set(ev.type, (seen.get(ev.type) || 0) + 1);
  if (ev.type === "session.updated") {
    log("session.updated");
    ws.send(JSON.stringify({
      type: "conversation.item.create",
      item: { type: "message", role: "user", content: [{ type: "input_text", text: "Announce yourself in one short line." }] }
    }));
    ws.send(JSON.stringify({ type: "response.create" }));
  } else if (ev.type === "response.output_audio_transcript.delta" || ev.type === "response.audio_transcript.delta") {
    process.stdout.write((ev.delta || ""));
  } else if (ev.type === "response.output_audio.delta" || ev.type === "response.audio.delta") {
    if (seen.get(ev.type) === 1) log("audio deltas streaming (base64 PCM16)");
  } else if (ev.type === "response.done") {
    console.log("");
    log("response.done status=" + (ev.response && ev.response.status));
    log("event counts: " + JSON.stringify([...seen]));
    try { ws.close(); } catch (e) {}
    setTimeout(() => process.exit(0), 250);
  } else if (ev.type === "error") {
    log("ERROR " + JSON.stringify(ev.error));
  }
};

ws.onerror = () => log("websocket error");
setTimeout(() => {
  log("timeout; counts: " + JSON.stringify([...seen]));
  process.exit(1);
}, 120000);
