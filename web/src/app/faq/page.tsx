'use client';

import { useState } from 'react';
import Link from 'next/link';

interface FAQItem {
  question: string;
  answer: string;
}

interface FAQSection {
  title: string;
  items: FAQItem[];
}

const faqData: FAQSection[] = [
  {
    title: 'General',
    items: [
      {
        question: 'What is Larun?',
        answer: 'Larun is a TinyML-powered platform for discovering exoplanets in NASA telescope data. It uses specialized neural network models to detect transit signals - the tiny dips in starlight caused by planets passing in front of their host stars.',
      },
      {
        question: 'Do I need astronomy expertise?',
        answer: 'No! Larun is designed to be accessible to everyone. The interface is straightforward - enter a target ID and click "Run Detection". Check out the User Guide for learning resources.',
      },
      {
        question: 'Can I really discover new exoplanets?',
        answer: 'Yes! Citizen scientists have contributed to numerous exoplanet discoveries. If Larun identifies a promising candidate, you can submit it to the TESS Follow-up Observing Program (TFOP) for professional validation.',
      },
    ],
  },
  {
    title: 'Detection',
    items: [
      {
        question: 'How accurate is the detection?',
        answer: 'The EXOPLANET-001 model achieves 82% accuracy on the validation set. This is sufficient to identify promising candidates, but all detections should be verified through the vetting process.',
      },
      {
        question: 'What is a false positive?',
        answer: 'A false positive is something that looks like a planetary transit but isn\'t. Common causes include eclipsing binary stars, stellar variability, and instrumental artifacts. The vetting process helps identify these.',
      },
      {
        question: 'What data sources does Larun support?',
        answer: 'Larun supports data from NASA\'s TESS (Transiting Exoplanet Survey Satellite) and Kepler missions. You can analyze any target with a TIC (TESS Input Catalog) or KIC (Kepler Input Catalog) identifier.',
      },
      {
        question: 'What is BLS periodogram analysis?',
        answer: 'Box Least Squares (BLS) is an algorithm that searches for periodic box-shaped dips in light curve data. It\'s specifically designed to detect the characteristic pattern of planetary transits and helps determine orbital periods.',
      },
    ],
  },
  {
    title: 'Data & Privacy',
    items: [
      {
        question: 'Where does the data come from?',
        answer: 'Larun fetches data from NASA\'s MAST (Mikulski Archive for Space Telescopes) archive. This includes observations from TESS, Kepler, and K2 missions. All data is publicly available.',
      },
      {
        question: 'Is my data private?',
        answer: 'Yes! All processing happens locally in your browser using TensorFlow.js. No telescope data is uploaded to our servers. Your analysis results are only stored if you choose to save them to your account.',
      },
      {
        question: 'Can I export my results?',
        answer: 'Yes, you can export your analysis results in multiple formats including CSV, JSON, and PDF reports. The reports are formatted for NASA submission compatibility.',
      },
    ],
  },
  {
    title: 'Pricing',
    items: [
      {
        question: 'Is Larun free?',
        answer: 'Yes! The free tier includes 5 analyses per month, which is enough for casual exploration. Paid tiers offer more analyses, API access, and additional features. See the Pricing page for details.',
      },
      {
        question: 'What\'s included in paid plans?',
        answer: 'Paid plans include more monthly analyses (50 for Monthly, unlimited for Annual), advanced AI models, priority processing, API access, and white-label report generation.',
      },
      {
        question: 'Can I cancel my subscription?',
        answer: 'Yes, you can cancel your subscription at any time. Your access will continue until the end of your current billing period.',
      },
    ],
  },
];

export default function FAQPage() {
  const [openItems, setOpenItems] = useState<Record<string, boolean>>({});

  const toggleItem = (sectionIndex: number, itemIndex: number) => {
    const key = `${sectionIndex}-${itemIndex}`;
    setOpenItems(prev => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <div className="min-h-screen bg-[#f1f3f4]">
      {/* Top Navigation */}
      <nav className="fixed top-0 left-0 right-0 h-16 bg-white border-b border-[#dadce0] flex items-center px-6 z-50">
        <div className="flex items-center gap-2">
          <Link href="/" className="text-[22px] font-medium text-[#202124] no-underline">
            Larun<span className="text-[#1a73e8]">.</span><span className="text-[#1a73e8]">Space</span>
          </Link>
        </div>

        <div className="flex-1 flex justify-center gap-2">
          <Link href="/" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Home
          </Link>
          <Link href="/dashboard" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Dashboard
          </Link>
          <Link href="/#pricing" className="px-4 py-2 text-[#3c4043] text-sm font-medium rounded hover:bg-[#f1f3f4] no-underline">
            Pricing
          </Link>
          <Link href="/faq" className="px-4 py-2 text-[#1a73e8] text-sm font-medium rounded no-underline">
            Docs
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <Link href="/dashboard" className="px-5 py-2 bg-[#202124] text-white text-sm font-medium rounded no-underline hover:bg-[#3c4043] transition-colors">
            Get Started
          </Link>
        </div>
      </nav>

      {/* Main Content */}
      <main className="pt-16 max-w-[800px] mx-auto px-8 py-8">
        <h1 className="text-[32px] font-normal text-[#202124] mb-4">Frequently Asked Questions</h1>
        <p className="text-[#3c4043] mb-8">Everything you need to know about Larun and exoplanet detection.</p>

        {faqData.map((section, sectionIndex) => (
          <div key={sectionIndex}>
            <h2 className="text-xl font-medium text-[#202124] mt-8 mb-4 pt-6 border-t border-[#dadce0]">
              {section.title}
            </h2>

            <div className="space-y-3">
              {section.items.map((item, itemIndex) => {
                const key = `${sectionIndex}-${itemIndex}`;
                const isOpen = openItems[key];

                return (
                  <div key={itemIndex} className="bg-white border border-[#dadce0] rounded-lg overflow-hidden">
                    <button
                      onClick={() => toggleItem(sectionIndex, itemIndex)}
                      className="w-full px-5 py-4 bg-transparent border-none text-left cursor-pointer text-[15px] font-medium text-[#202124] flex justify-between items-center hover:bg-[#f1f3f4] transition-colors"
                    >
                      <span>{item.question}</span>
                      <span className="text-[#5f6368] text-lg">{isOpen ? '−' : '+'}</span>
                    </button>
                    {isOpen && (
                      <div className="px-5 pb-4 text-[#3c4043] text-sm leading-relaxed">
                        <p>{item.answer}</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        ))}

        {/* Contact Section */}
        <div className="mt-12 p-6 bg-white border border-[#dadce0] rounded-lg text-center">
          <h3 className="text-lg font-medium text-[#202124] mb-2">Still have questions?</h3>
          <p className="text-[#5f6368] text-sm mb-4">
            Can&apos;t find the answer you&apos;re looking for? Check out our documentation or reach out to support.
          </p>
          <div className="flex justify-center gap-3">
            <Link
              href="/#features"
              className="px-5 py-2 bg-[#f1f3f4] text-[#202124] text-sm font-medium rounded no-underline hover:bg-[#e8eaed] transition-colors"
            >
              View Documentation
            </Link>
            <a
              href="https://github.com/Paddy1981/larun/issues"
              target="_blank"
              rel="noopener noreferrer"
              className="px-5 py-2 bg-[#202124] text-white text-sm font-medium rounded no-underline hover:bg-[#3c4043] transition-colors"
            >
              Contact Support
            </a>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 border-t border-[#dadce0] bg-white text-center">
        <div className="flex justify-center gap-6 mb-4">
          <a href="https://laruneng.com" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            laruneng.com
          </a>
          <a href="https://github.com/Paddy1981/larun" target="_blank" rel="noopener noreferrer" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            GitHub
          </a>
          <Link href="/#features" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            Documentation
          </Link>
          <Link href="/#pricing" className="text-[#5f6368] text-sm no-underline hover:text-[#202124]">
            Pricing
          </Link>
        </div>
        <p className="text-xs text-[#5f6368]">&copy; {new Date().getFullYear()} Larun.Space. All rights reserved.</p>
      </footer>
    </div>
  );
}
