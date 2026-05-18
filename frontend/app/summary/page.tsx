'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useMemo, useRef } from 'react';
import { useAppContext } from '../lib/store';
import { MatchRecord } from '../lib/schemas';
import {
  Download,
  Printer,
  Calendar,
  Shield,
  ArrowLeft,
  AlertCircle,
} from 'lucide-react';

// ─── Types ────────────────────────────────────────────────────────────────────

interface PersonTimeline {
  trackId: number;
  firstSeen: number;
  lastSeen: number;
  segments: ActionSegment[];
}

interface ActionSegment {
  action: string;
  from: number; // timestamp_sec start
  to: number;   // timestamp_sec end
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Format seconds as "0.07s" or "1m 03.20s" */
function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(2);
  return m > 0 ? `${m}m ${s.padStart(5, '0')}s` : `${s}s`;
}

/**
 * Convert a raw ML action label to its "-ing" (gerund) form.
 * Covers every label the action_recognition.py model can emit.
 */
function toGerund(action: string): string {
  // Known label map — covers all labels in action_recognition.py
  const map: Record<string, string> = {
    stand:       'standing',
    walk:        'walking',
    run:         'running',
    sit:         'sitting',
    fight:       'fighting',
    turned_away: 'turned away',
    riding_lmv:  'riding a vehicle',
    interacting: 'interacting',
    unknown:     'performing an unknown activity',
  };

  const key = action.toLowerCase().trim();
  if (map[key]) return map[key];

  // Already ends in "ing" — return as-is
  if (key.endsWith('ing')) return action;

  // Generic fallback: CVC doubling rule then add "ing"
  if (/[aeiou][b-df-hj-np-tv-z]{1}$/.test(key)) {
    return action + key[key.length - 1] + 'ing';
  }
  // Silent-e rule
  if (key.endsWith('e') && !key.endsWith('ee')) {
    return action.slice(0, -1) + 'ing';
  }
  return action + 'ing';
}


/**
 * Group a person's records into consecutive action segments.
 * e.g. [walking@0.07, walking@0.5, standing@1.0, walking@2.0]
 *   → [{action:'walking', from:0.07, to:0.5}, {action:'standing', from:1.0, to:1.0}, {action:'walking', from:2.0, to:2.0}]
 */
function buildSegments(records: MatchRecord[]): ActionSegment[] {
  if (records.length === 0) return [];

  const segments: ActionSegment[] = [];
  let cur: ActionSegment = {
    action: records[0].action ?? 'Unknown activity',
    from: records[0].timestamp_sec,
    to: records[0].timestamp_sec,
  };

  for (let i = 1; i < records.length; i++) {
    const action = records[i].action ?? 'Unknown activity';
    if (action === cur.action) {
      cur.to = records[i].timestamp_sec;
    } else {
      segments.push(cur);
      cur = { action, from: records[i].timestamp_sec, to: records[i].timestamp_sec };
    }
  }
  segments.push(cur);
  return segments;
}

/** Group all records by track_id and build timelines */
function groupByPerson(records: MatchRecord[]): PersonTimeline[] {
  const map = new Map<number, MatchRecord[]>();
  for (const rec of records) {
    if (!map.has(rec.track_id)) map.set(rec.track_id, []);
    map.get(rec.track_id)!.push(rec);
  }

  const timelines: PersonTimeline[] = [];
  map.forEach((recs, trackId) => {
    recs.sort((a, b) => a.timestamp_sec - b.timestamp_sec);
    timelines.push({
      trackId,
      firstSeen: recs[0].timestamp_sec,
      lastSeen: recs[recs.length - 1].timestamp_sec,
      segments: buildSegments(recs),
    });
  });

  timelines.sort((a, b) => a.firstSeen - b.firstSeen);
  return timelines;
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function SummaryPage() {
  const router = useRouter();
  const { analysisResult } = useAppContext();
  const printRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!analysisResult) router.push('/dashboard');
  }, [analysisResult, router]);

  if (!analysisResult) return null;

  const timelines = useMemo(() => {
    const matchedIds = analysisResult.matchedTrackIds ?? [];
    // If matched IDs exist, filter to only those tracks; otherwise, build from all records
    const records = (analysisResult.matchHistory ?? []).filter(r =>
      matchedIds.length > 0 ? matchedIds.includes(r.track_id) : true
    );
    return groupByPerson(records);
  }, [analysisResult.matchHistory, analysisResult.matchedTrackIds]);

  const noData = timelines.length === 0;

  return (
    <div style={{ minHeight: '100vh', background: '#f0ede8' }}>

      {/* ── Top Action Bar ── */}
      <div
        className="no-print"
        style={{
          background: '#fff',
          borderBottom: '1px solid #e8e4de',
          padding: '0.875rem 2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          position: 'sticky',
          top: 0,
          zIndex: 50,
        }}
      >
        <button
          onClick={() => router.push('/dashboard')}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.4rem',
            background: 'none',
            border: 'none',
            color: '#6b7280',
            cursor: 'pointer',
            fontSize: '0.875rem',
            fontWeight: 500,
          }}
        >
          <ArrowLeft size={16} /> Back to Dashboard
        </button>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button onClick={() => window.print()} style={actionBtnStyle}>
            <Download size={15} /> Export PDF
          </button>
          <button onClick={() => window.print()} style={actionBtnStyle}>
            <Printer size={15} /> Print
          </button>
        </div>
      </div>

      {/* ── Document ── */}
      <div style={{ padding: '2.5rem 1rem' }}>
        <div
          ref={printRef}
          style={{
            maxWidth: '680px',
            margin: '0 auto',
            background: '#fff',
            borderRadius: '12px',
            boxShadow: '0 4px 32px rgba(0,0,0,0.08)',
            overflow: 'hidden',
          }}
        >
          {/* Document Header */}
          <div
            style={{
              padding: '2.5rem 3rem 2rem',
              textAlign: 'center',
              borderBottom: '1px solid #f0ede8',
            }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                borderRadius: '50%',
                border: '2px solid #d4a017',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 1rem',
              }}
            >
              <Shield size={22} color="#d4a017" />
            </div>
            <p
              style={{
                fontSize: '0.7rem',
                letterSpacing: '0.15em',
                textTransform: 'uppercase',
                color: '#9ca3af',
                marginBottom: '0.5rem',
                fontWeight: 600,
              }}
            >
              Executive Brief
            </p>
            <h1
              style={{
                fontFamily: "'Playfair Display', serif",
                fontSize: '1.9rem',
                fontWeight: 700,
                color: '#1b2a4a',
                marginBottom: '0.75rem',
              }}
            >
              Case Summary
            </h1>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.4rem',
                color: '#6b7280',
                fontSize: '0.8rem',
              }}
            >
              <Calendar size={13} />
              <span>Generated on {analysisResult.generatedDate}</span>
            </div>
          </div>

          {/* Document Body */}
          <div style={{ padding: '2rem 3rem 3rem' }}>

            {/* Section 1 — Executive Overview */}
            <Section number={1} title="Executive Overview">
              <p style={{ color: '#374151', lineHeight: 1.7, fontSize: '0.9rem' }}>
                {analysisResult.executiveSummary}
              </p>
            </Section>

            {/* Section 2 — Person Detection Timeline */}
            <Section number={2} title="Person Detection Timeline">
              {noData ? (
                /* No detections */
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.75rem',
                    padding: '1rem 1.25rem',
                    background: '#fef9ec',
                    border: '1px solid #fcd34d',
                    borderRadius: '8px',
                  }}
                >
                  <AlertCircle size={16} color="#b45309" />
                  <p style={{ color: '#92400e', fontSize: '0.875rem', fontWeight: 500, margin: 0 }}>
                    No persons were detected in the submitted footage.
                  </p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
                  {timelines.map((person, idx) => (
                    <PersonBlock key={person.trackId} person={person} index={idx + 1} />
                  ))}
                </div>
              )}
            </Section>

            {/* Section 3 —Evidence*/}
            <Section number={3} title="Evidence">
              <ul
                style={{
                  paddingLeft: '1.2rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.5rem',
                  margin: 0,
                }}
              >
                {analysisResult.actionItems.map((item, i) => (
                  <li key={i} style={{ color: '#374151', fontSize: '0.875rem', lineHeight: 1.6 }}>
                    {item}
                  </li>
                ))}
              </ul>
            </Section>
          </div>

          {/* Document Footer */}
          <div
            style={{
              borderTop: '1px solid #f0ede8',
              padding: '1rem 3rem',
              display: 'flex',
              justifyContent: 'space-between',
              background: '#fafaf9',
            }}
          >
            <div>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af', margin: 0 }}>Generated by Corporate Legal AI</p>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af', margin: '2px 0 0' }}>
                Chain of Custody ID: {analysisResult.chainOfCustody}
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af', margin: 0 }}>Page 1 of 1</p>
              <p style={{ fontSize: '0.72rem', color: '#9ca3af', margin: '2px 0 0' }}>CONFIDENTIAL WORK PRODUCT</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Section wrapper (original design) ────────────────────────────────────────

function Section({
  number,
  title,
  children,
}: {
  number: number;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ marginBottom: '2rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
        <div
          style={{
            width: '3px',
            height: '20px',
            background: '#d4a017',
            borderRadius: '2px',
            flexShrink: 0,
          }}
        />
        <h2
          style={{
            fontFamily: "'Playfair Display', serif",
            fontSize: '1.1rem',
            fontWeight: 700,
            color: '#1b2a4a',
            margin: 0,
          }}
        >
          {number}. {title}
        </h2>
      </div>
      {children}
    </div>
  );
}

// ─── PersonBlock ──────────────────────────────────────────────────────────────

function PersonBlock({ person, index }: { person: PersonTimeline; index: number }) {
  return (
    <div style={{ display: 'flex', gap: '0.875rem' }}>
      {/* Gold diamond bullet (matches original FindingCard style) */}
      <div
        style={{
          width: '20px',
          height: '20px',
          borderRadius: '50%',
          background: '#1b2a4a',
          flexShrink: 0,
          marginTop: '2px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <div
          style={{
            width: '6px',
            height: '6px',
            background: '#d4a017',
            borderRadius: '1px',
            transform: 'rotate(45deg)',
          }}
        />
      </div>

      <div style={{ flex: 1 }}>
        {/* Person heading */}
        <h3
          style={{
            fontWeight: 700,
            fontSize: '0.92rem',
            color: '#1b2a4a',
            marginBottom: '0.2rem',
          }}
        >
          Person {index} (Track #{person.trackId})
        </h3>

        {/* First / Last line */}
        <p
          style={{
            fontSize: '0.82rem',
            color: '#4b5563',
            marginBottom: '0.6rem',
            lineHeight: 1.5,
          }}
        >
          First detected at{' '}
          <TimeChip>{fmt(person.firstSeen)}</TimeChip>
          {' '}— Last detected at{' '}
          <TimeChip>{fmt(person.lastSeen)}</TimeChip>
        </p>

        {/* Action segments */}
        <ul
          style={{
            paddingLeft: '1.1rem',
            margin: 0,
            display: 'flex',
            flexDirection: 'column',
            gap: '0.35rem',
          }}
        >
          {person.segments.map((seg, i) => (
            <li
              key={i}
              style={{ fontSize: '0.845rem', color: '#374151', lineHeight: 1.6 }}
            >
              {seg.from === seg.to ? (
                <>
                  <TimeChip>{fmt(seg.from)}</TimeChip>
                  {' — The person was '}
                  <span style={{ color: '#1b2a4a', fontWeight: 500 }}>
                    {toGerund(seg.action)}
                  </span>
                </>
              ) : (
                <>
                  {'The person was '}
                  <span style={{ color: '#1b2a4a', fontWeight: 600 }}>
                    {toGerund(seg.action)}
                  </span>
                  {' from '}
                  <TimeChip>{fmt(seg.from)}</TimeChip>
                  {' to '}
                  <TimeChip>{fmt(seg.to)}</TimeChip>
                </>
              )}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

// ─── TimeChip — small monospace timestamp pill ────────────────────────────────

function TimeChip({ children }: { children: React.ReactNode }) {
  return (
    <span
      style={{
        display: 'inline-block',
        fontFamily: 'monospace',
        fontSize: '0.78rem',
        fontWeight: 700,
        color: '#1b2a4a',
        background: '#f0ede8',
        border: '1px solid #d4a01733',
        borderRadius: '4px',
        padding: '0px 5px',
        lineHeight: '1.6',
      }}
    >
      {children}
    </span>
  );
}

// ─── Shared button style ──────────────────────────────────────────────────────

const actionBtnStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '0.4rem',
  padding: '0.5rem 1.1rem',
  border: '1px solid #d1cfc9',
  borderRadius: '8px',
  background: '#fff',
  color: '#374151',
  cursor: 'pointer',
  fontSize: '0.85rem',
  fontWeight: 500,
  transition: 'background 0.15s',
};
