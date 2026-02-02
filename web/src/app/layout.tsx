import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/lib/auth-provider";

export const metadata: Metadata = {
  title: "LARUN - Discover Exoplanets with AI",
  description: "Analyze NASA TESS and Kepler data using TinyML-powered transit detection. 81.8% accuracy. No PhD required.",
  keywords: ["exoplanet", "astronomy", "AI", "TinyML", "light curve", "transit detection", "Kepler", "TESS", "NASA"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="font-sans antialiased bg-white text-gray-900" style={{ fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
        <AuthProvider>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
