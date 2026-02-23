import { redirect } from 'next/navigation';

/**
 * /cloud/dashboard has been consolidated into /dashboard.
 * This redirect ensures all existing links keep working.
 */
export default function CloudDashboardRedirect() {
  redirect('/dashboard');
}
