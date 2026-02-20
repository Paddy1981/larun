import { NextResponse } from 'next/server';
import { TINYML_MODELS } from '@/lib/api-client';

export async function GET() {
  return NextResponse.json({ models: TINYML_MODELS });
}
