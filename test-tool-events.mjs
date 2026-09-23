const ws = new WebSocket("ws://127.0.0.1:8765/v1/realtime");
let callId = "";
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "session.update",
    session: {
      type: "realtime",
      instructions: "Use tools. Keep replies short.",
      tools: [{ type: "function", name: "get_time", description: "Get current time.", parameters: { type: "object", properties: {}, required: [] } }],
      tool_choice: "auto",
      audio: { input: { turn_detection: { type: "server_vad", interrupt_response: true } }, output: {} }
    }
  }));
};
ws.onmessage = (e) => {
  let ev; try { ev = JSON.parse(e.data); } catch { return; }
  if (ev.type === "session.updated") {
    ws.send(JSON.stringify({ type: "conversation.item.create", item: { type: "message", role: "user", content: [{ type: "input_text", text: "What time is it?" }] } }));
    ws.send(JSON.stringify({ type: "response.create" }));
  } else if (ev.type === "response.output_item.added" && ev.item && ev.item.type === "function_call") {
    callId = ev.item.call_id;
    console.log("output_item.added   item=" + ev.item.id + " call=" + ev.item.call_id);
  } else if (ev.type === "response.function_call_arguments.done") {
    console.log("arguments.done      item=" + ev.item_id + " args=" + ev.arguments);
  } else if (ev.type === "response.output_item.done" && ev.item && ev.item.type === "function_call") {
    console.log("output_item.done    item=" + ev.item.id + " call=" + ev.item.call_id + " args=" + ev.item.arguments);
  } else if (ev.type === "response.done") {
    console.log("response.done (call_id=" + callId + ")");
    try { ws.close(); } catch (e) {}
    setTimeout(() => process.exit(0), 200);
  }
};
setTimeout(() => process.exit(1), 45000);
