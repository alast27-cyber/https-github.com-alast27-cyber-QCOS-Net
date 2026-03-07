
import React, { useState, useEffect } from 'react';
import { KeyIcon, PlusIcon, InboxIcon, SendIcon, ShieldCheckIcon } from './Icons';
import { Button } from './Button';

interface SecureMailClientProps {
  userAddress: string;
}

const SecureMailClient: React.FC<SecureMailClientProps> = ({ userAddress }) => {
  const [activeTab, setActiveTab] = useState<'inbox' | 'compose' | 'sent'>('inbox');
  const [inboxMessages, setInboxMessages] = useState<any[]>([]);
  const [sentMessages, setSentMessages] = useState<any[]>([]);
  const [qkdStatus, setQkdStatus] = useState<'active' | 'inactive' | 'error'>('inactive');

  useEffect(() => {
    const fetchMail = async () => {
      try {
        const inboxRes = await fetch('/api/mail/inbox');
        const inboxData = await inboxRes.json();
        setInboxMessages(inboxData);

        const sentRes = await fetch('/api/mail/sent');
        const sentData = await sentRes.json();
        setSentMessages(sentData);
      } catch (e) {
        console.error(e);
      }
    };

    const checkQkdStatus = async () => {
      setQkdStatus('active');
    };

    fetchMail();
    checkQkdStatus();
  }, []);

  const handleSendMessage = async () => {
    const to = (document.getElementById('to') as HTMLInputElement).value;
    const subject = (document.getElementById('subject') as HTMLInputElement).value;
    const body = (document.getElementById('body') as HTMLTextAreaElement).value;

    try {
      await fetch('/api/mail/send', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ to, subject, body })
      });
      console.log('Sending message...');
      alert('Message sent securely via CHIPS network!');
      
      // Refresh sent messages
      const sentRes = await fetch('/api/mail/sent');
      const sentData = await sentRes.json();
      setSentMessages(sentData);
      
      setActiveTab('sent');
    } catch (e) {
      console.error(e);
      alert('Failed to send message.');
    }
  };

  const renderInbox = () => (
    <div className="space-y-2">
      {inboxMessages.length === 0 ? (
        <p className="text-cyan-600 text-center">Your inbox is empty.</p>
      ) : (
        inboxMessages.map((msg) => (
          <div key={msg.id} className={`p-3 rounded-lg border transition-colors cursor-pointer ${msg.isRead ? 'bg-black/20 border-cyan-800/50' : 'bg-cyan-900/40 border-cyan-700'} hover:bg-cyan-800/50`}>
            <div className="flex justify-between items-center text-cyan-200">
              <span className="font-semibold text-sm">{msg.sender}</span>
              <span className="text-xs text-cyan-400">{new Date(msg.timestamp).toLocaleString()}</span>
            </div>
            <div className="mt-1 text-white flex items-center">
              {msg.qkdSecured && <span title="Quantum Secured"><ShieldCheckIcon className="w-4 h-4 mr-2 text-green-400" /></span>}
              <span className="truncate text-sm">{msg.subject}</span>
            </div>
          </div>
        ))
      )}
    </div>
  );

  const renderCompose = () => (
    <div className="space-y-4">
      <div>
        <label htmlFor="to" className="block text-sm font-medium text-cyan-300">To:</label>
        <input type="text" id="to" className="mt-1 block w-full rounded-md bg-black/40 border-cyan-700 text-white shadow-sm focus:border-cyan-500 focus:ring-cyan-500 sm:text-sm p-2" />
      </div>
      <div>
        <label htmlFor="subject" className="block text-sm font-medium text-cyan-300">Subject:</label>
        <input type="text" id="subject" className="mt-1 block w-full rounded-md bg-black/40 border-cyan-700 text-white shadow-sm focus:border-cyan-500 focus:ring-cyan-500 sm:text-sm p-2" />
      </div>
      <div>
        <label htmlFor="body" className="block text-sm font-medium text-cyan-300">Body:</label>
        <textarea id="body" rows={6} className="mt-1 block w-full rounded-md bg-black/40 border-cyan-700 text-white shadow-sm focus:border-cyan-500 focus:ring-cyan-500 sm:text-sm p-2 resize-none"></textarea>
      </div>
      <Button
        onClick={handleSendMessage}
        title="Send the secure message"
        className="bg-cyan-600/50 hover:bg-cyan-700/70 border-cyan-500 text-white flex items-center w-full justify-center py-2"
      >
        <SendIcon className="w-5 h-5 mr-2" /> Send ChipsMail
      </Button>
    </div>
  );

  const renderSent = () => (
     <div className="space-y-2">
      {sentMessages.length === 0 ? (
        <p className="text-cyan-600 text-center">You haven't sent any messages yet.</p>
      ) : (
        sentMessages.map((msg) => (
          <div key={msg.id} className="p-3 rounded-lg border bg-black/20 border-cyan-800/50 hover:bg-cyan-800/50 transition-colors cursor-pointer">
            <div className="flex justify-between items-center text-cyan-200">
              <span className="font-semibold text-sm">To: {msg.recipient}</span>
              <span className="text-xs text-cyan-400">{new Date(msg.timestamp).toLocaleString()}</span>
            </div>
            <div className="mt-1 text-white flex items-center">
              {msg.qkdSecured && <span title="Quantum Secured"><ShieldCheckIcon className="w-4 h-4 mr-2 text-green-400" /></span>}
              <span className="truncate text-sm">{msg.subject}</span>
            </div>
          </div>
        ))
      )}
    </div>
  );

  return (
    <div className="p-2 h-full flex flex-col">
        <div className="flex justify-between items-center mb-4 flex-shrink-0">
            <div className="flex space-x-2">
            <Button
                onClick={() => setActiveTab('inbox')}
                title="View your inbox"
                className={`flex items-center text-white ${activeTab === 'inbox' ? 'bg-cyan-600/50 border-cyan-500' : 'bg-slate-700/50 border-slate-600 hover:bg-slate-600/50'}`}
            >
                <InboxIcon className="w-5 h-5 mr-2" /> Inbox
            </Button>
            <Button
                onClick={() => setActiveTab('compose')}
                title="Compose a new secure message"
                className={`flex items-center text-white ${activeTab === 'compose' ? 'bg-cyan-600/50 border-cyan-500' : 'bg-slate-700/50 border-slate-600 hover:bg-slate-600/50'}`}
            >
                <PlusIcon className="w-5 h-5 mr-2" /> Compose
            </Button>
            <Button
                onClick={() => setActiveTab('sent')}
                title="View sent messages"
                className={`flex items-center text-white ${activeTab === 'sent' ? 'bg-cyan-600/50 border-cyan-500' : 'bg-slate-700/50 border-slate-600 hover:bg-slate-600/50'}`}
            >
                <SendIcon className="w-5 h-5 mr-2" /> Sent
            </Button>
            </div>
            <div className="flex items-center text-sm">
            <KeyIcon className={`w-5 h-5 mr-2 ${qkdStatus === 'active' ? 'text-green-400 animate-pulse' : 'text-red-400'}`} />
            <span className={`${qkdStatus === 'active' ? 'text-green-300' : 'text-red-300'}`}>QKD: {qkdStatus.toUpperCase()}</span>
            </div>
        </div>
        <div className="mt-2 flex-grow overflow-y-auto pr-2 -mr-2">
            {activeTab === 'inbox' && renderInbox()}
            {activeTab === 'compose' && renderCompose()}
            {activeTab === 'sent' && renderSent()}
        </div>
    </div>
  );
};

export default SecureMailClient;
