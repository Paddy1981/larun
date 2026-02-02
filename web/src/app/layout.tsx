import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth-provider";

export const metadata: Metadata = {
  title: "AstroTinyML - Discover Exoplanets with AI",
  description: "TinyML-powered exoplanet detection achieving 81.8% accuracy. Upload light curve data from Kepler, TESS, or ground-based telescopes.",
  keywords: ["exoplanet", "astronomy", "AI", "TinyML", "light curve", "transit detection", "Kepler", "TESS"],
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
