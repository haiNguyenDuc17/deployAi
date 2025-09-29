import { environment } from '../environments/environment';

export interface SearchResult {
  title: string;
  url: string;
  snippet?: string;
  date?: string;
}

export interface ContextBundle {
  summary: string;
  bullets: string[];
  citations: { idx: number; url: string; title: string; date?: string }[];
}

const SEARCH_RESULT_LIMIT = 5;

export function shouldSearch(userInput: string): boolean {
  const text = userInput.toLowerCase();
  const intentKeywords = [
    'today', 'now', 'latest', 'mới nhất', 'gần đây', 'real-time', 'trực tiếp',
    'giá', 'price', 'news', 'tin tức', 'update', 'cập nhật', 'what happened',
    'current', 'hiện tại', 'recent', 'gần đây', 'btc price', 'bitcoin price',
    'give me', 'show me', 'tell me', 'what is', 'how much', 'bao nhiêu'
  ];
  return intentKeywords.some(k => text.includes(k));
}

export async function searchSerpApi(query: string): Promise<SearchResult[]> {
  if (!environment.SERPAPI_KEY) {
    console.warn('SERPAPI_KEY missing in environment');
    return [];
  }

  const params = new URLSearchParams({
    engine: 'google',
    q: query,
    num: String(SEARCH_RESULT_LIMIT),
    api_key: environment.SERPAPI_KEY,
  });

  try {
    const res = await fetch(`/serpapi/search.json?${params.toString()}`);
    if (!res.ok) {
      console.error(`SerpAPI error: ${res.status}`, await res.text());
      return [];
    }
    const data = await res.json();
    
    console.log('SerpAPI response:', data); // Debug log

    // Handle different response structures
    let organic = data.organic_results || [];
    
    // If no organic results, try answer_box or knowledge_graph
    if (organic.length === 0) {
      if (data.answer_box) {
        organic = [{
          title: data.answer_box.title || 'Answer',
          link: data.answer_box.link || '',
          snippet: data.answer_box.answer || data.answer_box.result || data.answer_box.snippet,
          date: data.answer_box.date
        }];
      } else if (data.knowledge_graph) {
        organic = [{
          title: data.knowledge_graph.title || 'Knowledge Graph',
          link: data.knowledge_graph.source?.link || '',
          snippet: data.knowledge_graph.description || data.knowledge_graph.type,
          date: data.knowledge_graph.date
        }];
      }
    }

    const results: SearchResult[] = organic.slice(0, SEARCH_RESULT_LIMIT).map((r: any) => ({
      title: r.title || 'Untitled',
      url: r.link || r.url || '',
      snippet: r.snippet || r.snippet_highlighted_words?.join(' ') || r.answer || r.description,
      date: r.date || r.published_date,
    }));

    console.log('Processed results:', results); // Debug log
    return results.filter(r => !!r.url && !!r.title);
  } catch (error) {
    console.error('SerpAPI search error:', error);
    return [];
  }
}

export function buildContext(results: SearchResult[], userQuery: string): ContextBundle | null {
  if (!results || results.length === 0) return null;

  const bullets: string[] = results.map((r, i) => {
    const snippet = r.snippet ? ` - ${r.snippet}` : '';
    const date = r.date ? ` (${r.date})` : '';
    return `${r.title}${snippet}${date} - ${r.url}`;
  });

  const summary = `Web search results for: "${userQuery}"`;
  const citations = results.map((r, i) => ({ idx: i + 1, url: r.url, title: r.title, date: r.date }));

  return { summary, bullets, citations };
}

export function formatContextMarkdown(ctx: ContextBundle): string {
  const lines: string[] = [];
  lines.push('Web search results:');
  lines.push(ctx.summary);
  lines.push('');
  ctx.bullets.forEach(b => lines.push(b));
  lines.push('');
  lines.push('Use this information to answer the user. Cite sources when relevant.');
  return lines.join('\n');
}


