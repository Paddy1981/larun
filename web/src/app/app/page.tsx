'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

const suggestedPrompts = [
  { icon: 'search', text: 'Search for transits in TIC 307210830' },
  { icon: 'clock', text: 'Is TOI-700 d in the habitable zone?' },
  { icon: 'chart', text: 'Analyze light curve for Kepler-11' },
  { icon: 'doc', text: 'Generate a report for my candidate planet' },
];

const quickActions = [
  { label: 'TIC lookup', prompt: 'TIC lookup for ' },
  { label: 'Upload light curve', action: 'upload' },
  { label: 'Run detection', prompt: 'Run detection on ' },
  { label: 'Check HZ', prompt: 'Check habitable zone for ' },
];

export default function ChatPage() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 200) + 'px';
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const generateResponse = (query: string): string => {
    const q = query.toLowerCase();
    if (q.includes('tic') || q.includes('search') || q.includes('transit')) {
      return "I'll search for transit signals in that target. The TinyML model is analyzing the light curve data from TESS...\n\nAnalysis complete! Found potential transit signal with:\n- Period: 3.42 days\n- Depth: 0.012%\n- SNR: 8.7\n\nThis appears to be a promising exoplanet candidate. Would you like me to run a detailed vetting analysis?";
    }
    if (q.includes('habitable') || q.includes('toi-700')) {
      return "TOI-700 d is indeed located in the habitable zone of its host star!\n\nKey facts:\n- Orbital period: 37.4 days\n- Equilibrium temperature: ~268K (-5C)\n- Size: 1.19 Earth radii\n- Receives ~86% of Earth's solar flux\n\nThis makes it one of the most Earth-like planets discovered by TESS.";
    }
    if (q.includes('kepler') || q.includes('light curve') || q.includes('analyze')) {
      return "Analyzing Kepler-11 light curve data...\n\nThis is a fascinating multi-planet system! I detected 6 transiting planets:\n- Kepler-11b: 1.97 Re, 10.3d period\n- Kepler-11c: 3.15 Re, 13.0d period\n- Kepler-11d: 3.43 Re, 22.7d period\n- Kepler-11e: 4.52 Re, 32.0d period\n- Kepler-11f: 2.61 Re, 46.7d period\n- Kepler-11g: 3.66 Re, 118.4d period\n\nAll six planets orbit closer to their star than Venus does to the Sun!";
    }
    if (q.includes('report')) {
      return "I can generate a comprehensive report for your candidate. Please provide:\n\n1. Target ID (TIC or KIC number)\n2. Detected period\n3. Transit depth\n\nOr upload your light curve data and I'll extract these parameters automatically.";
    }
    return "I'm Larun, your exoplanet discovery assistant. I can help you:\n\n- Search for transit signals in TESS/Kepler data\n- Analyze light curves for periodic signals\n- Calculate habitable zone boundaries\n- Generate discovery reports\n\nWhat would you like to explore?";
  };

  const sendMessage = () => {
    if (!inputValue.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
    }
    setIsTyping(true);

    // Simulate AI response
    setTimeout(() => {
      const response = generateResponse(userMessage.content);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);
      setIsTyping(false);
    }, 1000 + Math.random() * 1000);
  };

  const useSuggestedPrompt = (prompt: string) => {
    setInputValue(prompt);
    inputRef.current?.focus();
  };

  const startNewChat = () => {
    setMessages([]);
    setInputValue('');
    setSidebarOpen(false);
  };

  const formatMessage = (content: string) => {
    return content
      .replace(/`([^`]+)`/g, '<code class="bg-[#f1f3f4] px-1.5 py-0.5 rounded text-[13px] font-mono">$1</code>')
      .replace(/\n/g, '<br />');
  };

  return (
    <div className="flex h-screen bg-[#f1f3f4]">
      {/* Sidebar */}
      <aside className={`${sidebarOpen ? 'w-[280px]' : 'w-0'} bg-white border-r border-[#dadce0] flex flex-col transition-all overflow-hidden`}>
        <div className="p-4 border-b border-[#f1f3f4]">
          <button
            onClick={startNewChat}
            className="w-full py-3 px-4 bg-[#202124] hover:bg-[#3c4043] text-white rounded-lg text-sm font-medium flex items-center justify-center gap-2 transition-colors"
          >
            <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 5v14M5 12h14" />
            </svg>
            New Chat
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          <div className="mb-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[#5f6368] px-3 py-2">Today</p>
            <div className="space-y-0.5">
              {messages.length > 0 && (
                <div className="px-3 py-2.5 rounded-lg text-sm text-[#202124] bg-[#e8f0fe] cursor-pointer truncate">
                  {messages[0]?.content.substring(0, 30)}...
                </div>
              )}
            </div>
          </div>
          <div className="mb-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[#5f6368] px-3 py-2">Yesterday</p>
            <div className="space-y-0.5">
              <div className="px-3 py-2.5 rounded-lg text-sm text-[#3c4043] hover:bg-[#f1f3f4] cursor-pointer truncate">
                Transit search TIC 307210830
              </div>
              <div className="px-3 py-2.5 rounded-lg text-sm text-[#3c4043] hover:bg-[#f1f3f4] cursor-pointer truncate">
                Kepler-11 analysis
              </div>
            </div>
          </div>
          <div className="mb-4">
            <p className="text-[11px] font-semibold uppercase tracking-wider text-[#5f6368] px-3 py-2">Previous 7 Days</p>
            <div className="space-y-0.5">
              <div className="px-3 py-2.5 rounded-lg text-sm text-[#3c4043] hover:bg-[#f1f3f4] cursor-pointer truncate">
                TOI-700 habitable zone check
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-[#f1f3f4]">
          <div className="flex items-center gap-3 p-2 rounded-lg hover:bg-[#f1f3f4] cursor-pointer">
            <div className="w-8 h-8 bg-[#202124] text-white rounded-full flex items-center justify-center text-sm font-medium">
              U
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-[#202124] truncate">User</p>
              <p className="text-xs text-[#5f6368]">Explorer (Free)</p>
            </div>
            <svg className="w-4 h-4 text-[#5f6368]" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 9l6 6 6-6" />
            </svg>
          </div>
        </div>
      </aside>

      {/* Sidebar Overlay (mobile) */}
      {sidebarOpen && (
        <div className="fixed inset-0 bg-black/30 z-40 lg:hidden" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="h-16 bg-white border-b border-[#dadce0] flex items-center justify-between px-6">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden p-2 hover:bg-[#f1f3f4] rounded-lg text-[#3c4043]"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 12h18M3 6h18M3 18h18" />
              </svg>
            </button>
            <Link href="/" className="text-[22px] font-medium text-[#202124]">
              Larun<span className="text-[#5f6368]">.</span>
            </Link>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/auth/login" className="px-4 py-2 text-sm font-medium text-[#3c4043] hover:bg-[#f1f3f4] rounded">
              Sign In
            </Link>
            <Link href="/#pricing" className="px-4 py-2 text-sm font-medium text-[#3c4043] hover:bg-[#f1f3f4] rounded flex items-center gap-1.5">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
              </svg>
              Upgrade
            </Link>
            <Link href="/#features" className="px-4 py-2 text-sm font-medium text-[#3c4043] hover:bg-[#f1f3f4] rounded">
              Docs
            </Link>
          </div>
        </header>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-[800px] mx-auto">
            {messages.length === 0 ? (
              /* Welcome Screen */
              <div className="text-center py-20">
                <h1 className="text-5xl font-medium text-[#202124] mb-3">
                  Larun<span className="text-[#5f6368]">.</span>
                </h1>
                <p className="text-lg text-[#5f6368] mb-12">What would you like to discover today?</p>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-[640px] mx-auto">
                  {suggestedPrompts.map((prompt, index) => (
                    <button
                      key={index}
                      onClick={() => useSuggestedPrompt(prompt.text)}
                      className="bg-white border border-[#dadce0] rounded-xl p-4 text-left flex items-start gap-3 hover:border-[#1a73e8] hover:shadow-md transition-all"
                    >
                      <svg className="w-5 h-5 text-[#5f6368] flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        {prompt.icon === 'search' && <><circle cx="11" cy="11" r="8" /><path d="M21 21l-4.35-4.35" /></>}
                        {prompt.icon === 'clock' && <><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></>}
                        {prompt.icon === 'chart' && <><path d="M3 3v18h18" /><path d="M18 17l-5-10-4 8-3-4" /></>}
                        {prompt.icon === 'doc' && <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M16 13H8M16 17H8" /></>}
                      </svg>
                      <span className="text-sm text-[#3c4043]">{prompt.text}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* Messages */
              <div className="space-y-6">
                {messages.map((message) => (
                  <div key={message.id} className="flex gap-4">
                    <div className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 ${
                      message.role === 'assistant' ? 'bg-[#202124] text-white' : 'bg-[#1a73e8] text-white'
                    }`}>
                      {message.role === 'assistant' ? (
                        <svg className="w-4.5 h-4.5" fill="currentColor" viewBox="0 0 24 24">
                          <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
                        </svg>
                      ) : (
                        <span className="text-sm font-medium">U</span>
                      )}
                    </div>
                    <div
                      className="flex-1 text-[15px] leading-relaxed text-[#3c4043]"
                      dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
                    />
                  </div>
                ))}

                {/* Typing Indicator */}
                {isTyping && (
                  <div className="flex gap-4">
                    <div className="w-9 h-9 rounded-full bg-[#202124] text-white flex items-center justify-center flex-shrink-0">
                      <svg className="w-4.5 h-4.5" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" />
                      </svg>
                    </div>
                    <div className="flex items-center gap-1 py-2">
                      <div className="w-2 h-2 bg-[#5f6368] rounded-full animate-pulse" style={{ animationDelay: '0ms' }} />
                      <div className="w-2 h-2 bg-[#5f6368] rounded-full animate-pulse" style={{ animationDelay: '200ms' }} />
                      <div className="w-2 h-2 bg-[#5f6368] rounded-full animate-pulse" style={{ animationDelay: '400ms' }} />
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 md:p-6 bg-[#f1f3f4]">
          <div className="max-w-[800px] mx-auto">
            <div className="bg-white border border-[#dadce0] rounded-xl flex items-end p-3 focus-within:border-[#1a73e8] focus-within:shadow-[0_0_0_2px_rgba(26,115,232,0.15)] transition-all">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask Larun anything about exoplanets..."
                rows={1}
                className="flex-1 border-none outline-none resize-none text-[15px] leading-relaxed max-h-[200px] bg-transparent"
              />
              <div className="flex gap-2 ml-3">
                <button
                  onClick={() => setShowUploadModal(true)}
                  className="p-1.5 text-[#5f6368] hover:bg-[#f1f3f4] hover:text-[#3c4043] rounded"
                  title="Upload light curve"
                >
                  <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                  </svg>
                </button>
                <button
                  onClick={sendMessage}
                  disabled={!inputValue.trim()}
                  className="bg-[#202124] hover:bg-[#3c4043] text-white p-2 rounded-lg disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <svg className="w-4.5 h-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path d="M22 2L11 13" />
                    <path d="M22 2l-7 20-4-9-9-4 20-7z" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex flex-wrap gap-2 mt-3 justify-center">
              {quickActions.map((action, index) => (
                <button
                  key={index}
                  onClick={() => action.action === 'upload' ? setShowUploadModal(true) : useSuggestedPrompt(action.prompt || '')}
                  className="text-xs text-[#5f6368] px-2.5 py-1 bg-white border border-[#dadce0] rounded-full hover:border-[#1a73e8] hover:text-[#1a73e8] transition-colors"
                >
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Upload Modal */}
      {showUploadModal && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl p-8 max-w-md w-full">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-medium text-[#202124]">Upload Light Curve</h3>
              <button onClick={() => setShowUploadModal(false)} className="text-[#5f6368] hover:text-[#202124]">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="border-2 border-dashed border-[#dadce0] rounded-xl p-12 text-center mb-4 cursor-pointer hover:border-[#1a73e8] transition-colors">
              <svg className="w-12 h-12 text-[#5f6368] mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <p className="text-[#5f6368] text-sm">Drag & drop or click to upload</p>
              <p className="text-[#5f6368] text-xs mt-2">Supports FITS, CSV, or TXT files</p>
            </div>

            <button
              onClick={() => setShowUploadModal(false)}
              className="w-full py-2.5 text-sm text-[#3c4043] hover:bg-[#f1f3f4] rounded-lg transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
