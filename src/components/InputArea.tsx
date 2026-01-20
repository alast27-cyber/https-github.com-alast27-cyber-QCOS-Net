
import React, { useRef, useState } from 'react';
import { Button } from './Button';
import { Attachment } from '../types';
import { Paperclip, X, Send, Image as ImageIcon } from 'lucide-react';

interface InputAreaProps {
  onSend: (text: string, attachments: Attachment[]) => void;
  isLoading: boolean;
  isImageMode: boolean;
}

export const InputArea: React.FC<InputAreaProps> = ({ onSend, isLoading, isImageMode }) => {
  const [text, setText] = useState('');
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSend = () => {
    if ((!text.trim() && attachments.length === 0) || isLoading) return;
    onSend(text, attachments);
    setText('');
    setAttachments([]);
    // Reset height
    if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      // Basic validation
      if (!file.type.startsWith('image/')) {
        alert('Only image files are supported currently.');
        return;
      }

      const reader = new FileReader();
      reader.onload = (event) => {
        const base64 = event.target?.result as string;
        const newAttachment: Attachment = {
          file,
          previewUrl: URL.createObjectURL(file),
          mimeType: file.type,
          base64: base64,
        };
        setAttachments((prev) => [...prev, newAttachment]);
      };
      reader.readAsDataURL(file);
    }
    // Reset input
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeAttachment = (index: number) => {
    setAttachments((prev) => prev.filter((_, i) => i !== index));
  };

  const handleTextareaInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
      const target = e.currentTarget;
      target.style.height = 'auto';
      target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
      setText(target.value);
  }

  return (
    <div className="p-4 bg-slate-900 border-t border-slate-800">
      <div className="max-w-4xl mx-auto flex flex-col gap-3">
        {/* Attachments Preview */}
        {attachments.length > 0 && (
          <div className="flex gap-2 overflow-x-auto pb-2">
            {attachments.map((att, idx) => (
              <div key={idx} className="relative group flex-shrink-0">
                <img
                  src={att.previewUrl}
                  alt="preview"
                  className="h-20 w-20 object-cover rounded-md border border-slate-700"
                />
                <button
                  onClick={() => removeAttachment(idx)}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Input Bar */}
        <div className="flex items-end gap-2 bg-slate-800 p-2 rounded-xl border border-slate-700 focus-within:ring-2 focus-within:ring-blue-500/50 transition-all">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded-lg transition-colors"
            title="Add image"
            disabled={isLoading || isImageMode} // Disable upload in image gen mode for simplicity
          >
            {isImageMode ? <ImageIcon size={20} className="opacity-50" /> : <Paperclip size={20} />}
          </button>
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/*"
            onChange={handleFileChange}
          />
          
          <textarea
            ref={textareaRef}
            value={text}
            onChange={handleTextareaInput}
            onKeyDown={handleKeyDown}
            placeholder={isImageMode ? "Describe the image you want to generate..." : "Send a message..."}
            className="flex-1 bg-transparent text-slate-100 placeholder-slate-500 resize-none py-2 px-1 focus:outline-none max-h-48 overflow-y-auto"
            rows={1}
            disabled={isLoading}
          />
          
          <Button
            onClick={handleSend}
            disabled={(!text.trim() && attachments.length === 0) || isLoading}
            size="sm"
            className="mb-0.5 rounded-lg"
          >
            <Send size={18} />
          </Button>
        </div>
        <div className="text-xs text-center text-slate-500">
           {isImageMode ? "Gemini 2.5 Flash Image Model" : "Gemini can make mistakes. Check important info."}
        </div>
      </div>
    </div>
  );
};
