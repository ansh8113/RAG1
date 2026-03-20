"use client";

import { useState } from "react";

export default function Home() {

  const [question, setQuestion] = useState("");

  const [messages, setMessages] = useState<
    { role: string; text: string }[]
  >([]);

  const askQuestion = async () => {

  try {

    const res = await fetch("http://127.0.0.1:8000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();

    setMessages((prev) => [
      ...prev,
      { role: "user", text: question },
      { role: "assistant", text: data.answer },
    ]);

    setQuestion("");

  } catch (error) {

    console.error("Error:", error);

  }
};

  return (
    <div className="p-10">

      <h1 className="text-3xl font-bold mb-5">
        Advanced RAG Chatbot
      </h1>

      {messages.map((msg, index) => (
        <div key={index} className="mb-3">
          <b>{msg.role}:</b> {msg.text}
        </div>
      ))}

      <input
        className="border p-2 w-full"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
      />

      <button
        onClick={askQuestion}
        className="bg-blue-500 text-white px-4 py-2 mt-3"
      >
        Ask
      </button>

  </div>
  );
  }