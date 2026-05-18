/**
 * api.ts
 * All real API calls to the FastAPI backend at http://localhost:8000
 * Using native fetch — no axios needed for this scale.
 */

import { AnalysisResult } from './schemas';

const BASE_URL = 'http://localhost:8000';

// ─── Helpers ──────────────────────────────────────────────────────────────────

async function apiPost<T>(path: string, body: object, token?: string): Promise<T> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`);
  }
  return data as T;
}

async function apiGet<T>(path: string, token?: string): Promise<T> {
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE_URL}${path}`, { method: 'GET', headers });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`);
  }
  return data as T;
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

export interface RegisterPayload {
  email: string;
  password: string;
  full_name: string;
}

export interface RegisterResponse {
  id: string;
  email: string;
  full_name: string | null;
}

export interface LoginPayload {
  email: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface MeResponse {
  id: string;
  email: string;
  full_name: string | null;
}

/** Register a new user. Returns the created user. */
export async function registerUser(payload: RegisterPayload): Promise<RegisterResponse> {
  return apiPost<RegisterResponse>('/auth/register', payload);
}

/** Login — returns a JWT access token. */
export async function loginUser(payload: LoginPayload): Promise<LoginResponse> {
  return apiPost<LoginResponse>('/auth/login', payload);
}

/** Fetch the currently logged-in user profile using the stored JWT. */
export async function getMe(token: string): Promise<MeResponse> {
  return apiGet<MeResponse>('/auth/me', token);
}

// ─── Process (Video + Image Analysis) ─────────────────────────────────────────

interface BackendProcessResult {
  found: boolean;
  total_match_frames: number;
  first_appearance_sec: number | null;
  last_appearance_sec: number | null;
  best_match: {
    frame: number;
    timestamp_sec: number;
    track_id: number;
    similarity: number;
    action: string | null;
  } | null;
  matched_track_ids: number[];
  output_video_id: string | null;
  match_history: Array<{
    frame: number;
    timestamp_sec: number;
    track_id: number;
    similarity: number;
    action: string | null;
  }>;
}

/**
 * Upload video + image to the backend /process endpoint.
 * Maps the raw ProcessResult into the frontend's AnalysisResult shape.
 */
export async function analyzeEvidence(
  video: File,
  image: File
): Promise<AnalysisResult> {
  if (!(video instanceof File) || !(image instanceof File)) {
    throw new Error('Both image and video must be valid File objects.');
  }

  const formData = new FormData();
  formData.append('video', video, video.name || 'video.mp4');
  formData.append('image', image, image.name || 'image.jpg');

  const res = await fetch(`${BASE_URL}/process`, {
    method: 'POST',
    body: formData,
    // Do NOT set Content-Type header — browser sets it with boundary automatically
  });

  const raw = await res.json();
  if (!res.ok) {
    throw new Error(raw.detail || `Processing failed: ${res.status}`);
  }

  const result: BackendProcessResult = raw;

  // ── Map to frontend AnalysisResult ──────────────────────────────────────────
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  const videoLabel = video.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');
  const imageLabel = image.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');

  const personFound = result.found;
  const bestSeen = result.best_match;

  // Executive summary
  let executiveSummary = '';
  if (personFound && bestSeen) {
    const sim = (bestSeen.similarity * 100).toFixed(1);
    const first = result.first_appearance_sec?.toFixed(2) ?? 'N/A';
    const last = result.last_appearance_sec?.toFixed(2) ?? 'N/A';
    executiveSummary =
      `Analysis of the submitted footage (${videoLabel}) against the reference image (${imageLabel}) ` +
      `confirmed the target person was identified in ${result.total_match_frames} frame(s) ` +
      `with a peak similarity of ${sim}%. ` +
      `First appearance at ${first}s, last at ${last}s. ` +
      `Track IDs involved: ${result.matched_track_ids.join(', ')}.`;
  } else {
    executiveSummary =
      `Analysis of the submitted footage (${videoLabel}) against the reference image (${imageLabel}) ` +
      `did not detect the target person in any frame. ` +
      `The pipeline processed the full video using YOLOv8 + DeepSORT without a confirmed match.`;
  }

  // Build findings from match_history (show up to 3 most notable)
  const topMatches = [...result.match_history]
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 3);

  const findings: AnalysisResult['findings'] = topMatches.length > 0
    ? topMatches.map((m, i) => ({
        id: `f${i + 1}`,
        title: `Match at ${m.timestamp_sec.toFixed(2)}s — Track #${m.track_id}`,
        description:
          `Person detected at frame ${m.frame} (${m.timestamp_sec.toFixed(2)}s) with similarity score ` +
          `${(m.similarity * 100).toFixed(1)}%.` +
          (m.action ? ` Detected action: ${m.action}.` : ''),
        tag: `VID ${m.timestamp_sec.toFixed(1)}s`,
        sourceType: 'video' as const,
      }))
    : [
        {
          id: 'f1',
          title: 'No Match Found',
          description:
            'The pipeline completed without detecting the target person in any frame of the submitted video.',
          tag: 'VID-SCAN',
          sourceType: 'video' as const,
        },
      ];

  // Action items
  const actionItems = personFound
    ? [
        `Review annotated output video for Track ID(s): ${result.matched_track_ids.join(', ')}.`,
        `Verify match frames ${result.first_appearance_sec?.toFixed(2)}s – ${result.last_appearance_sec?.toFixed(2)}s with legal counsel.`,
        'Preserve original media files as chain-of-custody evidence.',
      ]
    : [
        'Consider providing a higher-resolution or clearer reference image.',
        'Ensure the subject appears prominently in the submitted video.',
        'Retry with adjusted detection threshold if needed.',
      ];

  return {
    caseTitle: `${imageLabel} — Track Vision Analysis`,
    caseRef: `TV-${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`,
    generatedDate: dateStr,
    chainOfCustody: result.output_video_id ?? 'N/A',
    executiveSummary,
    findings,
    actionItems,
    matchHistory: result.match_history,
    matchedTrackIds: result.matched_track_ids,
  };
}
