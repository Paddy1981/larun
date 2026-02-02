import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth-provider";

export const metadata: Metadata = {
  title: "LARUN - Discover Exoplanets with AI",
  description: "AI-powered light curve analysis for exoplanet discovery. Upload your astronomical data and let machine learning find hidden worlds.",
  keywords: ["exoplanet", "astronomy", "AI", "light curve", "transit detection", "machine learning"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="font-sans antialiased bg-white text-slate-900">
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
