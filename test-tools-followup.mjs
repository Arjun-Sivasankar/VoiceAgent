const url = "ws://127.0.0.1:8765/v1/realtime";
const ws = new WebSocket(url);
const t0 = Date.now();
const log = (m) => console.log(((Date.now() - t0) / 1000).toFixed(1) + "s " + m);
let callId = "";
let done = 0;
let stage = 1;
let needContinue = false;

function finish(code, message) {
  log(message);
  try { ws.close(); } catch (e) {}
  setTimeout(() => process.exit(code), 250);
}

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

function ask(text) {
  ws.send(JSON.stringify({ type: "conversation.item.create", item: { type: "message", role: "user", content: [{ type: "input_text", text }] } }));
  ws.send(JSON.stringify({ type: "response.create" }));
}

ws.onmessage = (e) => {
  let ev;
  try { ev = JSON.parse(e.data); } catch { return; }
  const type = ev.type || "";
  if (type === "session.updated") {
    log("stage 1: ask the time (tool turn)");
    ask("What time is it right now?");
  } else if (type === "response.output_item.added" && ev.item && ev.item.type === "function_call") {
    callId = ev.item.call_id;
    log("tool call: " + ev.item.name);
  } else if (type === "response.function_call_arguments.done") {
    ws.send(JSON.stringify({ type: "conversation.item.create", item: { type: "function_call_output", call_id: callId || ev.call_id, output: new Date().toString() } }));
    needContinue = true;
  } else if (type === "response.output_audio_transcript.done" || type === "response.audio_transcript.done") {
    log("assistant: " + JSON.stringify(ev.transcript));
  } else if (type === "response.done") {
    done += 1;
    log("response.done #" + done + " status=" + (ev.response && ev.response.status));
    if (needContinue) {
      needContinue = false;
      log("continuing after tool");
      ws.send(JSON.stringify({ type: "response.create" }));
      return;
    }
    if (done === 2 && stage === 1) {
      stage = 2;
      log("stage 2: normal follow-up turn (no tool)");
      setTimeout(() => ask("Thanks. What is two plus two?"), 400);
    } else if (done === 3 && stage === 2) {
      stage = 3;
      log("stage 3: another follow-up turn");
      setTimeout(() => ask("And what is the square root of nine?"), 400);
    } else if (done >= 4) {
      finish(0, "ALL TURNS OK - no provider 400");
    }
  } else if (type === "error") {
    finish(2, "SERVER ERROR: " + JSON.stringify(ev.error));
  }
};

setTimeout(() => finish(1, "timeout stage=" + stage + " done=" + done), 120000);
