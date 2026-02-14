import Header from '@/components/Header';
import Link from 'next/link';

export default function CloudLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      {/* Main Site Header */}
      <Header />

      {/* Cloud Content */}
      {children}

      {/* Main Site Footer */}
      <footer className="py-12 bg-[#f1f3f4] border-t border-[#dadce0]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-bold text-[#202124]">Larun<span className="text-[#1a73e8]">.</span></span>
              <span className="text-sm text-[#5f6368]">AstroTinyML</span>
            </Link>

            {/* Links */}
            <div className="flex items-center gap-6 text-sm text-[#5f6368]">
              <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="hover:text-[#202124] transition-colors">
                Larun Engineering
              </a>
              <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="hover:text-[#202124] transition-colors">
                GitHub
              </a>
              <Link href="#" className="hover:text-[#202124] transition-colors">
                Privacy
              </Link>
              <Link href="#" className="hover:text-[#202124] transition-colors">
                Terms
              </Link>
            </div>

            {/* Copyright */}
            <p className="text-sm text-[#5f6368]">
              &copy; {new Date().getFullYear()} Larun Engineering. All rights reserved.
            </p>
          </div>
        </div>
      </footer>
    </>
  );
}
