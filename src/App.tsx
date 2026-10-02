import React, { useState } from 'react';
import {
  ShieldAlert,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  FileCode,
  Terminal,
  Copy,
  ExternalLink,
  Lock,
  GitBranch,
  Layers,
  Sparkles
} from 'lucide-react';

interface BugItem {
  id: string;
  category: 'Security' | 'Crash' | 'Compatibility' | 'Robustness';
  severity: 'Critical' | 'High' | 'Medium' | 'Low';
  title: string;
  why: string;
  fix: string;
}

const BUGS_RESOLVED: BugItem[] = [
  {
    id: 'sec-key',
    category: 'Security',
    severity: 'Critical',
    title: 'Hardcoded API Key in Source Code',
    why: 'The Semantic Scholar API key was embedded directly as a string literal (API_KEY = "s2k-hlv..."). Committing this to Git permanently leaks your credential, risks quota exhaustion, and exposes you to billing/abuse.',
    fix: 'Switched to python-dotenv & os.getenv("S2_API_KEY"). Added .env* to .gitignore and provided a template in .env.example. Warned to revoke and rotate the exposed key immediately.',
  },
  {
    id: 'cross-platform-open',
    category: 'Compatibility',
    severity: 'High',
    title: 'Windows-Only explorer.exe Subprocess Call',
    why: 'subprocess.run(["explorer.exe", output_path]) crashes with FileNotFoundError on Linux, macOS, and container environments like Docker or Cloud Run.',
    fix: 'Replaced with Python standard library webbrowser.open(Path(output_path).resolve().as_uri()) with a graceful fallback.',
  },
  {
    id: 'cache-json-pickling',
    category: 'Security',
    severity: 'High',
    title: 'Arbitrary Code Execution Risk & Cache Invalidation with Pickle',
    why: 'Pickle files (.pkl) can trigger arbitrary code execution if modified, cannot easily be inspected, and sorted(json_data.items()) crashed when json_data contained nested lists or dictionaries.',
    fix: 'Switched to clean JSON disk caching (.json) with json.dumps(params, sort_keys=True, default=str) for collision-free and type-safe SHA-256 cache keys.',
  },
  {
    id: 'none-citation-crash',
    category: 'Crash',
    severity: 'High',
    title: 'TypeError: must be real number, not NoneType on New Papers',
    why: 'Semantic Scholar API often returns citationCount: null for newly indexed or preprint manuscripts. When p.get("citationCount") is None, math.sqrt(None) or min(citations_list) raises fatal TypeErrors.',
    fix: 'Sanitized citationCount and year: p.get("citationCount") if p.get("citationCount") is not None else (p.get("citations") or 0) and ensured integer conversions with fallbacks.',
  },
  {
    id: 'fragile-index-seed',
    category: 'Robustness',
    severity: 'Medium',
    title: 'Hardcoded i == 0 Seed Node Assumption in Graph Builder',
    why: 'The graph algorithms checked (i == 0 and parsed[j]["id"] in seed_refs_ids). If deduplication reordered or altered node lists, index 0 would no longer reliably be the seed paper.',
    fix: 'Replaced index assumptions with explicit parsed[i]["is_seed"] and parsed[j]["is_seed"] booleans.',
  },
  {
    id: 'tfidf-vocab-crash',
    category: 'Crash',
    severity: 'Medium',
    title: 'Empty Vocabulary Crash in TfidfVectorizer',
    why: 'If a small cluster of papers has missing abstracts or short titles whose terms are all in the English stopwords list, scikit-learn throws ValueError: empty vocabulary; perhaps the documents only contain stop words.',
    fix: 'Wrapped fit_transform in a try-except block, gracefully falling back to identity matrix similarity so graph generation never fails.',
  },
  {
    id: 'url-query-parsing',
    category: 'Robustness',
    severity: 'Medium',
    title: 'Unparsed Full URLs for DOIs and arXiv Links',
    why: 'Users frequently paste full URLs like https://arxiv.org/abs/2301.12345 or https://doi.org/10.1145/3613424. The original code only checked clean.startswith("10.") or clean.startswith("arxiv:").',
    fix: 'Added regex extractors for arXiv and DOI URLs to automatically extract the canonical IDs.',
  },
  {
    id: 'html-script-injection',
    category: 'Security',
    severity: 'Medium',
    title: 'Premature </script> Termination in HTML Template',
    why: 'If an academic paper title or abstract contains </script> (common in security or web research), injecting raw JSON into the HTML breaks the script block and corrupts the visualization.',
    fix: 'Created safe_json_for_script() that escapes </ sequences as <\\/ before embedding.',
  },
];

export default function App() {
  const [copiedKey, setCopiedKey] = useState(false);
  const [copiedEnv, setCopiedEnv] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'bugs' | 'git' | 'code'>('overview');

  const copyToClipboard = (text: string, type: 'key' | 'env') => {
    navigator.clipboard.writeText(text);
    if (type === 'key') {
      setCopiedKey(true);
      setTimeout(() => setCopiedKey(false), 2000);
    } else {
      setCopiedEnv(true);
      setTimeout(() => setCopiedEnv(false), 2000);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col selection:bg-indigo-500 selection:text-white">
      {/* Top Warning Banner */}
      <div className="bg-amber-500/10 border-b border-amber-500/30 px-6 py-3 flex items-center justify-between text-amber-300 text-xs sm:text-sm">
        <div className="flex items-center gap-2 font-medium">
          <ShieldAlert className="w-4 h-4 shrink-0 text-amber-400" />
          <span>
            <strong>Immediate Security Action Required:</strong> Since the key <code className="bg-amber-950/60 px-1.5 py-0.5 rounded text-amber-200">s2k-hlv...</code> was exposed, make sure to revoke/rotate it on Semantic Scholar!
          </span>
        </div>
        <a
          href="https://www.semanticscholar.org/product/api"
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 hover:underline text-amber-200 font-semibold shrink-0 ml-4"
        >
          S2 API Dashboard <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>

      {/* Main Header */}
      <header className="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-40 px-6 py-4 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Literature Graph & S2 API Tools
              <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                Git-Ready & Patched
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Clean environment configuration, bug audits, and Git sanitization
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 p-1 rounded-lg text-xs font-medium">
          <button
            onClick={() => setActiveTab('overview')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'overview'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Overview & Run
          </button>
          <button
            onClick={() => setActiveTab('bugs')}
            className={`px-3 py-1.5 rounded-md transition-colors flex items-center gap-1.5 ${
              activeTab === 'bugs'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <span>Bug Fixes</span>
            <span className="w-5 h-5 rounded-full bg-slate-800 flex items-center justify-center text-[10px] font-bold text-indigo-300">
              {BUGS_RESOLVED.length}
            </span>
          </button>
          <button
            onClick={() => setActiveTab('git')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'git'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Git & .env Setup
          </button>
          <button
            onClick={() => setActiveTab('code')}
            className={`px-3 py-1.5 rounded-md transition-colors ${
              activeTab === 'code'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Python File
          </button>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-6xl w-full mx-auto p-6 space-y-8">
        {activeTab === 'overview' && (
          <div className="space-y-6">
            {/* Quick Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 relative overflow-hidden">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <ShieldCheck className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">API Key Sanitized</h3>
                    <p className="text-xs text-slate-400 mt-0.5">Removed from source code</p>
                  </div>
                </div>
                <div className="mt-4 text-xs text-slate-300 bg-slate-950/80 rounded-lg p-2.5 font-mono border border-slate-800/80">
                  S2_API_KEY loaded via .env
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 relative overflow-hidden">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-lg bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                    <GitBranch className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">.gitignore Protected</h3>
                    <p className="text-xs text-slate-400 mt-0.5">Locks caches & secrets</p>
                  </div>
                </div>
                <div className="mt-4 text-xs text-slate-300 bg-slate-950/80 rounded-lg p-2.5 font-mono border border-slate-800/80">
                  .env*, .cache_s2/, *.pkl
                </div>
              </div>

              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 relative overflow-hidden">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    <CheckCircle2 className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white">8 Patches Applied</h3>
                    <p className="text-xs text-slate-400 mt-0.5">Crashes & edge cases fixed</p>
                  </div>
                </div>
                <div className="mt-4 text-xs text-slate-300 bg-slate-950/80 rounded-lg p-2.5 font-mono border border-slate-800/80">
                  Cross-platform, null guards, URL regex
                </div>
              </div>
            </div>

            {/* How to Run Section */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <Terminal className="w-4 h-4 text-indigo-400" />
                How to Run Your Safe Script
              </h2>

              <div className="space-y-4">
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
                    <span>1. Install dependencies:</span>
                  </div>
                  <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-indigo-300 overflow-x-auto">
                    pip install -r requirements.txt
                  </pre>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
                    <span>2. Create your local <code className="text-indigo-300">.env</code> file (safe, never pushed to Git):</span>
                  </div>
                  <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-emerald-300 overflow-x-auto">
                    S2_API_KEY=your_rotated_semantic_scholar_api_key_here
                  </pre>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-400 font-medium">
                    <span>3. Execute the script:</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div>
                      <span className="text-[11px] text-slate-400 block mb-1">Interactive prompt:</span>
                      <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-200">
                        python literature_graph.py
                      </pre>
                    </div>
                    <div>
                      <span className="text-[11px] text-slate-400 block mb-1">Direct query / DOI / arXiv URL:</span>
                      <pre className="bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-200">
                        python literature_graph.py "https://arxiv.org/abs/1706.03762"
                      </pre>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Key Features of the Patched Script */}
            <div className="bg-slate-900/50 border border-slate-800/80 rounded-xl p-6">
              <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-violet-400" />
                Script Highlights & Capabilities
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300 leading-relaxed">
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
                  <strong className="text-indigo-300 block mb-1">Dual-Graph Visualization</strong>
                  Separates bleeding-edge contemporary literature (contemporaries, direct siblings) from foundational roots (ancestors, landmark mechanisms).
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
                  <strong className="text-indigo-300 block mb-1">Conference Tier Classification</strong>
                  Automated badge taxonomy for Top Architecture (ISCA, MICRO, ASPLOS, HPCA), Top Systems/CS (SOSP, OSDI, NSDI, NeurIPS), Premier Systems, and Preprints.
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
                  <strong className="text-indigo-300 block mb-1">Cross-Platform Launcher</strong>
                  Works on Windows, macOS, and Linux without crashing on Windows-specific executables.
                </div>
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-800/60">
                  <strong className="text-indigo-300 block mb-1">Robust Rate-Limiting</strong>
                  Automatic exponential backoff, Retry-After header parsing, and fallback when run in unauthenticated public mode.
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'bugs' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-white">Comprehensive Bug & Vulnerability Audit</h2>
                <p className="text-xs text-slate-400">Detailed rationale for each patch applied to your test file</p>
              </div>
              <span className="text-xs px-2.5 py-1 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-medium">
                {BUGS_RESOLVED.length} Resolved Issues
              </span>
            </div>

            <div className="space-y-3">
              {BUGS_RESOLVED.map((bug, index) => (
                <div
                  key={bug.id}
                  className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 hover:border-slate-700 transition-colors"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-slate-500">#{index + 1}</span>
                      <h3 className="text-sm font-bold text-white">{bug.title}</h3>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                          bug.severity === 'Critical'
                            ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                            : bug.severity === 'High'
                            ? 'bg-amber-500/10 text-amber-400 border border-amber-500/20'
                            : 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                        }`}
                      >
                        {bug.severity}
                      </span>
                      <span className="text-[10px] font-medium px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                        {bug.category}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3 pt-3 border-t border-slate-800/80 text-xs leading-relaxed">
                    <div className="bg-rose-950/20 border border-rose-900/30 rounded-lg p-3">
                      <div className="font-semibold text-rose-300 mb-1 flex items-center gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5" />
                        WHY this is an issue:
                      </div>
                      <p className="text-rose-200/80">{bug.why}</p>
                    </div>

                    <div className="bg-emerald-950/20 border border-emerald-900/30 rounded-lg p-3">
                      <div className="font-semibold text-emerald-300 mb-1 flex items-center gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        HOW it was patched:
                      </div>
                      <p className="text-emerald-200/80">{bug.fix}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'git' && (
          <div className="space-y-6">
            {/* Git Best Practices Guide */}
            <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-6 space-y-4">
              <h2 className="text-base font-semibold text-white flex items-center gap-2">
                <GitBranch className="w-4 h-4 text-emerald-400" />
                Git Hygiene & Secret Prevention
              </h2>
              <p className="text-xs text-slate-300 leading-relaxed">
                Before running <code className="bg-slate-800 px-1 py-0.5 rounded text-indigo-300">git add .</code> or pushing to GitHub/GitLab, ensure your repository adheres to these security rules:
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="bg-slate-950 border border-slate-800 p-4 rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-emerald-300 flex items-center gap-1.5">
                      <Lock className="w-3.5 h-3.5" /> Updated .gitignore
                    </span>
                    <button
                      onClick={() => copyToClipboard('.env*\n!.env.example\n.cache_s2/\n*.pkl\ngraph*.html\n__pycache__/\n*.py[cod]', 'env')}
                      className="text-slate-400 hover:text-white"
                      title="Copy rules"
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <pre className="text-slate-400 text-[11px] font-mono leading-relaxed bg-slate-900/50 p-2.5 rounded border border-slate-800/80">
{`.env*
!.env.example
.cache_s2/
*.pkl
graph.html
graph_*.html
__pycache__/
*.py[cod]`}
                  </pre>
                  <p className="text-[11px] text-slate-400">
                    Prevents credentials, API response caches, and rendered graphs from being committed.
                  </p>
                </div>

                <div className="bg-slate-950 border border-slate-800 p-4 rounded-lg space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-indigo-300 flex items-center gap-1.5">
                      <FileCode className="w-3.5 h-3.5" /> Template .env.example
                    </span>
                    <button
                      onClick={() => copyToClipboard('S2_API_KEY="your_semantic_scholar_api_key_here"', 'key')}
                      className="text-slate-400 hover:text-white"
                      title="Copy template"
                    >
                      <Copy className="w-3.5 h-3.5" />
                    </button>
                  </div>
                  <pre className="text-slate-400 text-[11px] font-mono leading-relaxed bg-slate-900/50 p-2.5 rounded border border-slate-800/80">
{`# Semantic Scholar API Key
# https://www.semanticscholar.org/product/api
S2_API_KEY="your_api_key_here"`}
                  </pre>
                  <p className="text-[11px] text-slate-400">
                    Safe to commit to Git! Shows team members or collaborators what environment variables are needed.
                  </p>
                </div>
              </div>

              {/* Git Command Checklist */}
              <div className="bg-slate-950 border border-slate-800/90 rounded-lg p-4 space-y-2">
                <span className="text-xs font-semibold text-slate-200">Terminal Verification Steps:</span>
                <pre className="text-xs font-mono text-slate-300 space-y-1">
                  <div><span className="text-slate-500"># 1. Verify git is ignoring .env and cache:</span></div>
                  <div className="text-indigo-300">git status --ignored</div>
                  <div className="text-slate-500 mt-2"># 2. Stage safe files:</div>
                  <div className="text-indigo-300">git add literature_graph.py requirements.txt .gitignore .env.example</div>
                  <div className="text-slate-500 mt-2"># 3. Check diff to confirm NO secrets are present:</div>
                  <div className="text-indigo-300">git diff --cached</div>
                </pre>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'code' && (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-semibold text-white flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-indigo-400" />
                  literature_graph.py
                </h2>
                <p className="text-xs text-slate-400">The sanitized, bug-fixed script ready for Git</p>
              </div>
              <button
                onClick={() => {
                  navigator.clipboard.writeText('Saved as /literature_graph.py in workspace root.');
                  alert('The complete script is saved to /literature_graph.py in your project!');
                }}
                className="px-3 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-colors flex items-center gap-1.5"
              >
                <CheckCircle2 className="w-3.5 h-3.5" />
                Script Saved in Workspace
              </button>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs font-mono text-slate-300 space-y-2">
              <div className="flex items-center justify-between text-slate-400 pb-2 border-b border-slate-800">
                <span>Location: /literature_graph.py</span>
                <span>Language: Python 3.9+</span>
              </div>
              <p className="text-slate-400 text-xs">
                Key imports: <code className="text-indigo-300">requests</code>, <code className="text-indigo-300">numpy</code>, <code className="text-indigo-300">scikit-learn</code>, <code className="text-indigo-300">python-dotenv</code>, <code className="text-indigo-300">webbrowser</code>.
              </p>
              <div className="bg-slate-950 p-4 rounded-lg border border-slate-800 text-[11px] leading-relaxed text-slate-300">
                <span className="text-emerald-400"># Securely reading the key with fallback to unauthenticated mode:</span>
                <br />
                <code className="text-slate-200">API_KEY = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY") or ""</code>
                <br />
                <code className="text-slate-200">client = S2Client(API_KEY)</code>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
