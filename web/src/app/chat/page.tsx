'use client';

import { useState, useRef, useEffect } from 'react';
import Link from 'next/link';

interface Message {
  id: string;
  role: 'user' | 'bot';
  content: string;
}

const suggestions = [
  'Analyze TIC 307210830 for exoplanets',
  'What is a transit method?',
  'Kepler-90 exoplanets',
  'Explain BLS analysis',
];

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'signup'>('login');
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messageIdRef = useRef(0);

  const scrollToBottom = () => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const generateResponse = (query: string): string => {
    const q = query.toLowerCase();

    if (q.includes('tic') || q.includes('analyze')) {
      const ticMatch = query.match(/TIC\s*(\d+)/i);
      const ticId = ticMatch ? ticMatch[1] : '307210830';
      return `**Analyzing TIC ${ticId}...**\n\nI found the following data:\n\n• **Star Type:** G-type main sequence\n• **Magnitude:** 11.2\n• **Distance:** 156 parsecs\n\n**Transit Detection Results:**\n• **Candidates Found:** 2 potential signals\n• **Period 1:** 3.42 days (confidence: 94%)\n• **Period 2:** 7.81 days (confidence: 78%)\n\nWould you like me to run a BLS periodogram for more detailed analysis?`;
    }

    if (q.includes('transit')) {
      return `**The Transit Method**\n\nThe transit method detects exoplanets by measuring the tiny dip in a star's brightness when a planet passes in front of it.\n\n**Key Points:**\n• Brightness dip is typically 0.01% to 1%\n• Requires precise photometry\n• Can determine planet size and orbital period\n• Works best for edge-on planetary systems\n\nLARUN uses TinyML models to detect these subtle signals in light curve data.`;
    }

    if (q.includes('bls') || q.includes('periodogram')) {
      return `**Box Least Squares (BLS) Periodogram**\n\nBLS is an algorithm specifically designed to detect periodic box-shaped dips in time series data - perfect for finding transiting exoplanets!\n\n**How it works:**\n1. Tests many possible orbital periods\n2. For each period, fits a box-shaped transit model\n3. Calculates how well each model matches the data\n4. Peaks in the periodogram indicate likely transit signals\n\nLARUN combines BLS with machine learning for higher accuracy.`;
    }

    if (q.includes('kepler-90') || q.includes('kepler 90')) {
      return `**Kepler-90 System**\n\nKepler-90 is a Sun-like star with **8 confirmed exoplanets** - tied with our Solar System for the most known planets!\n\n**Known Planets:**\n• Kepler-90b: 1.31 Earth radii, 7.0 day orbit\n• Kepler-90c: 1.18 Earth radii, 8.7 day orbit\n• Kepler-90d: 2.88 Earth radii, 59.7 day orbit\n• Kepler-90e: 2.67 Earth radii, 91.9 day orbit\n• Kepler-90f: 2.89 Earth radii, 124.9 day orbit\n• Kepler-90g: 8.13 Earth radii, 210.6 day orbit\n• Kepler-90h: 11.32 Earth radii, 331.6 day orbit\n• Kepler-90i: 1.32 Earth radii, 14.4 day orbit\n\nThe 8th planet (Kepler-90i) was discovered using machine learning!`;
    }

    return `I can help you with:\n\n• **Analyzing targets** - e.g., "Analyze TIC 307210830"\n• **Transit detection** - Finding exoplanet candidates\n• **BLS analysis** - Periodogram searches\n• **Astronomical concepts** - Transit method, light curves, etc.\n\nWhat would you like to explore?`;
  };

  const formatMessage = (text: string) => {
    return text
      .replace(/\n/g, '<br />')
      .replace(/`([^`]+)`/g, '<code class="bg-black/30 px-1.5 py-0.5 rounded text-sm font-mono">$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  };

  const sendMessage = (text?: string) => {
    const content = text || inputValue.trim();
    if (!content) return;

    messageIdRef.current += 1;
    const userMessage: Message = {
      id: `msg-${messageIdRef.current}`,
      role: 'user',
      content,
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);

    setTimeout(() => {
      const response = generateResponse(content);
      messageIdRef.current += 1;
      const botMessage: Message = {
        id: `msg-${messageIdRef.current}`,
        role: 'bot',
        content: response,
      };
      setMessages(prev => [...prev, botMessage]);
      setIsTyping(false);
    }, 1500);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-white flex flex-col" style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif" }}>
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 bg-[#12121a] border-b border-[#2a2a3a]">
        <Link href="/" className="flex items-center gap-2.5 text-white font-semibold text-lg no-underline">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
              <circle cx="12" cy="12" r="10"/>
              <circle cx="12" cy="12" r="3"/>
              <path d="M12 2v4M12 18v4M2 12h4M18 12h4"/>
            </svg>
          </div>
          <span>LARUN Chat</span>
        </Link>
        <div className="flex items-center gap-3">
          <Link href="/dashboard" className="px-4 py-2 rounded-lg text-sm font-medium text-[#a0a0b0] hover:bg-[#1a1a24] hover:text-white transition-all">
            Dashboard
          </Link>
          <button onClick={clearChat} className="px-4 py-2 rounded-lg text-sm font-medium text-[#a0a0b0] hover:bg-[#1a1a24] hover:text-white transition-all">
            New Chat
          </button>
          <button
            onClick={() => setShowAuthModal(true)}
            className="px-4 py-2 rounded-lg text-sm font-medium text-white transition-all"
            style={{ background: '#6366f1' }}
            onMouseOver={(e) => e.currentTarget.style.background = '#818cf8'}
            onMouseOut={(e) => e.currentTarget.style.background = '#6366f1'}
          >
            Sign In
          </button>
        </div>
      </header>

      {/* Chat Container */}
      <div className="flex-1 flex flex-col max-w-[900px] w-full mx-auto">
        {messages.length === 0 ? (
          /* Welcome Screen */
          <div className="flex-1 flex flex-col items-center justify-center px-5 py-10 text-center">
            <div
              className="w-20 h-20 rounded-[20px] flex items-center justify-center mb-6 text-4xl"
              style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}
            >
              🔭
            </div>
            <h1 className="text-[1.75rem] font-semibold mb-3">Welcome to LARUN Chat</h1>
            <p className="text-[#a0a0b0] max-w-[500px] mb-8">
              Your AI assistant for exoplanet detection. Ask me to analyze any star, search for transits, or explain astronomical concepts.
            </p>
            <div className="flex flex-wrap gap-2.5 justify-center max-w-[600px]">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  onClick={() => sendMessage(suggestion)}
                  className="px-4 py-2.5 rounded-full text-sm text-[#a0a0b0] border border-[#2a2a3a] bg-[#1a1a24] hover:bg-[#12121a] hover:border-[#6366f1] hover:text-white transition-all cursor-pointer"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Messages Area */
          <div ref={messagesRef} className="flex-1 overflow-y-auto p-5 flex flex-col gap-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 max-w-[85%] animate-[fadeIn_0.3s_ease] ${message.role === 'user' ? 'self-end flex-row-reverse' : ''}`}
              >
                <div
                  className={`w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-sm ${
                    message.role === 'bot' ? '' : 'bg-[#1a1a24] border border-[#2a2a3a]'
                  }`}
                  style={message.role === 'bot' ? { background: 'linear-gradient(135deg, #6366f1, #a855f7)' } : {}}
                >
                  {message.role === 'bot' ? '🔭' : '👤'}
                </div>
                <div
                  className={`py-3 px-4 rounded-2xl leading-relaxed ${
                    message.role === 'bot'
                      ? 'bg-[#1e1e2e] border border-[#2a2a3a] rounded-bl-sm'
                      : 'rounded-br-sm'
                  }`}
                  style={message.role === 'user' ? { background: '#6366f1' } : {}}
                  dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
                />
              </div>
            ))}

            {/* Typing Indicator */}
            {isTyping && (
              <div className="flex gap-3 max-w-[85%]">
                <div
                  className="w-9 h-9 rounded-full flex items-center justify-center flex-shrink-0 text-sm"
                  style={{ background: 'linear-gradient(135deg, #6366f1, #a855f7)' }}
                >
                  🔭
                </div>
                <div className="py-3 px-4 bg-[#1e1e2e] border border-[#2a2a3a] rounded-2xl rounded-bl-sm">
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-[#606070] rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="w-2 h-2 bg-[#606070] rounded-full animate-bounce" style={{ animationDelay: '200ms' }} />
                    <span className="w-2 h-2 bg-[#606070] rounded-full animate-bounce" style={{ animationDelay: '400ms' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="px-5 py-4 pb-6 bg-[#0a0a0f] border-t border-[#2a2a3a]">
        <div className="max-w-[900px] mx-auto flex gap-3 items-end">
          <div className="flex-1 bg-[#12121a] border border-[#2a2a3a] rounded-xl flex items-end p-1 focus-within:border-[#6366f1] transition-colors">
            <textarea
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask about any star or exoplanet..."
              rows={1}
              className="flex-1 bg-transparent border-none text-white text-[15px] py-2.5 px-3 resize-none min-h-[44px] max-h-[150px] outline-none placeholder-[#606070]"
              style={{ fontFamily: 'inherit' }}
            />
          </div>
          <button
            onClick={() => sendMessage()}
            disabled={!inputValue.trim()}
            className="w-11 h-11 rounded-[10px] text-white flex items-center justify-center flex-shrink-0 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:enabled:scale-105"
            style={{ background: '#6366f1' }}
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Auth Modal */}
      {showAuthModal && (
        <div className="fixed inset-0 bg-black/80 z-[1000] flex items-center justify-center">
          <div className="bg-[#12121a] border border-[#2a2a3a] rounded-2xl p-6 max-w-[400px] w-[90%]">
            <div className="flex justify-between items-center mb-5">
              <h3 className="text-xl font-medium">{authMode === 'login' ? 'Sign In' : 'Create Account'}</h3>
              <button onClick={() => setShowAuthModal(false)} className="text-[#a0a0b0] hover:text-white text-2xl leading-none">
                &times;
              </button>
            </div>

            <div className="flex flex-col gap-3">
              <button className="w-full py-3 px-4 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg text-white flex items-center justify-center gap-2 hover:bg-[#12121a] transition-colors">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
                </svg>
                Continue with GitHub
              </button>

              <div className="flex items-center gap-3 my-1">
                <div className="flex-1 h-px bg-[#2a2a3a]" />
                <span className="text-xs text-[#606070]">OR</span>
                <div className="flex-1 h-px bg-[#2a2a3a]" />
              </div>

              <input
                type="email"
                placeholder="Email"
                className="py-3 px-4 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg text-white text-[15px] outline-none focus:border-[#6366f1]"
              />
              <input
                type="password"
                placeholder="Password"
                className="py-3 px-4 bg-[#1a1a24] border border-[#2a2a3a] rounded-lg text-white text-[15px] outline-none focus:border-[#6366f1]"
              />
              <button
                className="w-full py-3 rounded-lg text-white font-medium transition-all"
                style={{ background: '#6366f1' }}
              >
                {authMode === 'login' ? 'Sign In' : 'Create Account'}
              </button>

              <p className="text-center text-sm text-[#606070] mt-2">
                {authMode === 'login' ? "Don't have an account? " : 'Already have an account? '}
                <button
                  onClick={() => setAuthMode(authMode === 'login' ? 'signup' : 'login')}
                  className="text-[#6366f1] hover:underline"
                >
                  {authMode === 'login' ? 'Sign Up' : 'Sign In'}
                </button>
              </p>
            </div>
          </div>
        </div>
      )}

      <style jsx global>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
