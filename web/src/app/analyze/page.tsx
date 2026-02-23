import { redirect } from 'next/navigation'

// Legacy /analyze — merged into /cloud/analyze (TIC ID + file upload + colour indices)
export default function AnalyzePage() {
  redirect('/cloud/analyze')
}
