import { AnalysisResult } from './schemas';

// Simulates an AI backend analysis of uploaded files
export async function analyzeEvidence(
  video: File,
  image: File
): Promise<AnalysisResult> {
  // Simulate backend processing delay (3-5 seconds)
  await new Promise((resolve) => setTimeout(resolve, 4000));

  const now = new Date();
  const dateStr = now.toLocaleDateString('en-US', {
    month: 'long',
    day: 'numeric',
    year: 'numeric',
  });

  // Generate result based on the actual file names
  const videoName = video.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');
  const imageName = image.name.replace(/\.[^.]+$/, '').replace(/[-_]/g, ' ');

  return {
    caseTitle: 'State v. Doe',
    caseRef: 'CV-2023-0892',
    generatedDate: dateStr,
    chainOfCustody: '9A8F-2B4C-95E1',
    executiveSummary:
      `This brief synthesizes critical findings extracted from submitted surveillance footage (${videoName}) ` +
      `and photographic evidence (${imageName}) pertaining to the incident on review. ` +
      'The automated analysis indicates significant discrepancies between the defendant\'s initial statement ' +
      'and the verifiable timeline established by the raw media.',
    findings: [
      {
        id: 'f1',
        title: 'Timeline Discrepancy: Entry Point',
        description:
          'The footage clearly demonstrates the suspect entering the secondary premises at exactly 2:34 AM, ' +
          'contradicting the sworn testimony claiming presence at the location from 3:00 AM onwards.',
        tag: 'VID-01 02:34',
        sourceType: 'video',
      },
      {
        id: 'f2',
        title: 'Unidentified Vehicle Identification',
        description:
          'Image enhancement reveals an unidentified blue sedan parked in the loading zone. ' +
          'While the primary license plate is partially obscured by glare, partial EXIF metadata and ' +
          'geometric analysis suggest a local state plate structure.',
        tag: 'IMG-04',
        sourceType: 'image',
      },
      {
        id: 'f3',
        title: 'Altered Documentation Suspected',
        description:
          'Review of the provided ledger snapshots indicates inconsistent ink density and anomalous alignment ' +
          'in the entries dated Oct 10th - 12th, suggesting potential post-facto modification.',
        tag: 'IMG-12',
        sourceType: 'image',
      },
    ],
    actionItems: [
      'Subpoena complete traffic camera logs for the intersection of 5th and Sycamore between 2:00 AM and 4:00 AM.',
      'Request original physical copies of the ledger spanning October for forensic ink analysis.',
      'Interview secondary witness listed in initial report regarding the blue sedan.',
    ],
  };
}
