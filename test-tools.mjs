const url = "ws://127.0.0.1:8765/v1/realtime";
const ws = new WebSocket(url);
const t0 = Date.now();
const log = (m) => console.log(((Date.now() - t0) / 1000).toFixed(1) + "s " + m);
let callId = "";
let done = 0;

ws.onopen = () => {
  log("open");
  ws.send(JSON.stringify({
    type: "session.update",
    session: {
      type: "realtime",
      instructions: "You are a helpful assistant. Use tools when needed. Keep replies short.",
      tools: [{
        type: "function",
        name: "get_time",
        description: "Get the current local date and time.",
        parameters: { type: "object", properties: {}, required: [] }
      }],
      tool_choice: "auto",
      audio: { input: { turn_detection: { type: "server_vad", interrupt_response: true } }, output: {} }
    }
  }));
};

ws.onmessage = (e) => {
  let ev;
  try { ev = JSON.parse(e.data); } catch { return; }
  if (ev.type === "session.updated") {
    log("session.updated -> asking the time");
    ws.send(JSON.stringify({ type: "conversation.item.create", item: { type: "message", role: "user", content: [{ type: "input_text", text: "What time is it right now?" }] } }));
    ws.send(JSON.stringify({ type: "response.create" }));
  } else if (ev.type === "response.output_item.added" && ev.item && ev.item.type === "function_call") {
    callId = ev.item.call_id;
    log("tool call requested: " + ev.item.name + " call_id=" + callId);
  } else if (ev.type === "response.function_call_arguments.done") {
    log("args done: " + ev.arguments);
    ws.send(JSON.stringify({
      type: "conversation.item.create",
      item: { type: "function_call_output", call_id: callId || ev.call_id, output: new Date().toString() }
    }));
    ws.send(JSON.stringify({ type: "response.create" }));
  } else if (ev.type === "response.output_audio_transcript.delta" || ev.type === "response.audio_transcript.delta") {
    process.stdout.write(ev.delta || "");
  } else if (ev.type === "response.output_audio_transcript.done" || ev.type === "response.audio_transcript.done") {
    log("assistant said: " + JSON.stringify(ev.transcript));
  } else if (ev.type === "response.done") {
    done += 1;
    log("response.done #" + done);
    if (done >= 2) setTimeout(() => process.exit(0), 300);
  } else if (ev.type === "error") {
    log("ERROR " + JSON.stringify(ev.error));
    process.exit(2);
  }
};

setTimeout(() => { log("timeout done=" + done); process.exit(1); }, 90000);
