import React, { useState, useEffect } from "react";

const QAPINotifier: React.FC = () => {
  const [activeNode, setActiveNode] = useState<string | null>(null);

  // Poll for incoming QAPI calls from the Mesh (Polling fallback for serverless)
  useEffect(() => {
    let lastEventTimestamp = Date.now();

    const checkForEvents = async () => {
      try {
        const response = await fetch("/api/qapi/events");
        if (response.ok) {
          const events = await response.json();
          if (events && events.length > 0) {
            const latestEvent = events[events.length - 1];
            if (latestEvent.timestamp > lastEventTimestamp) {
              lastEventTimestamp = latestEvent.timestamp;
              setActiveNode(latestEvent.origin); // e.g., "CHIPS_BROWSER"
              setTimeout(() => setActiveNode(null), 3000); // Reset after 3 seconds
            }
          }
        }
      } catch (err) {
        console.error("QAPI Polling Error:", err);
      }
    };

    const intervalId = setInterval(checkForEvents, 1000);
    return () => clearInterval(intervalId);
  }, []);

  if (!activeNode) return null;

  return (
    <div className="fixed top-5 right-5 qapi-active z-[9999]">
      <span className="blink animate-pulse pr-2 text-xl">●</span> QAPI
      Handshake: {activeNode} Entangled
    </div>
  );
};

export default QAPINotifier;
