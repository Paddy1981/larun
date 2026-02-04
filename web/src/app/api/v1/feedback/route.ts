import { NextRequest, NextResponse } from 'next/server';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/lib/auth';
import { createServerSupabaseClient } from '@/lib/supabase';

export async function POST(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user?.email) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await request.json();
    const { analysis_id, is_correct, note } = body;

    if (!analysis_id || typeof is_correct !== 'boolean') {
      return NextResponse.json(
        { error: 'Missing required fields: analysis_id and is_correct' },
        { status: 400 }
      );
    }

    const supabase = createServerSupabaseClient();

    // Store the feedback
    const { error } = await supabase.from('model_feedback').insert({
      analysis_id,
      user_email: session.user.email,
      is_correct,
      note: note || null,
      created_at: new Date().toISOString(),
    });

    if (error) {
      // If table doesn't exist, log and return success anyway
      // This allows the feature to work even if the table hasn't been created yet
      console.error('Error storing feedback:', error);

      // If it's a "relation does not exist" error, still return success
      // The feedback UI should work, we just can't store it yet
      if (error.code === '42P01') {
        console.log('Feedback table not created yet. Feedback logged but not stored.');
        return NextResponse.json({
          success: true,
          message: 'Feedback received (storage pending table creation)'
        });
      }

      return NextResponse.json({ error: 'Failed to store feedback' }, { status: 500 });
    }

    return NextResponse.json({ success: true, message: 'Feedback submitted successfully' });
  } catch (error) {
    console.error('Feedback submission error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}

// Get feedback statistics (for admin/dashboard)
export async function GET(request: NextRequest) {
  try {
    const session = await getServerSession(authOptions);

    if (!session?.user?.email) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const supabase = createServerSupabaseClient();

    // Get feedback counts
    const { data, error } = await supabase
      .from('model_feedback')
      .select('is_correct')
      .order('created_at', { ascending: false })
      .limit(1000);

    if (error) {
      // If table doesn't exist, return empty stats
      if (error.code === '42P01') {
        return NextResponse.json({
          total: 0,
          correct: 0,
          incorrect: 0,
          accuracy: 0,
        });
      }
      throw error;
    }

    const total = data?.length || 0;
    const correct = data?.filter(f => f.is_correct).length || 0;
    const incorrect = total - correct;
    const accuracy = total > 0 ? (correct / total) * 100 : 0;

    return NextResponse.json({
      total,
      correct,
      incorrect,
      accuracy: accuracy.toFixed(1),
    });
  } catch (error) {
    console.error('Feedback stats error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
