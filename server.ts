import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

// Lazy GoogleGenAI client initialization
let aiClient: GoogleGenAI | null = null;
function getAI(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is not defined");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

async function callGenerateContentWithFallback(
  ai: GoogleGenAI,
  options: {
    model: string;
    contents: any;
    config?: any;
  }
) {
  const primaryModel = options.model;
  const backupModel = "gemini-3.1-flash-lite";

  const modelsToTry = [primaryModel, backupModel];
  let lastError: any = null;

  for (const currentModel of modelsToTry) {
    let attempts = 2; // Try up to 2 times with a small delay
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        console.log(`Executing generateContent with model: ${currentModel} (Attempt ${attempt}/${attempts})`);
        const response = await ai.models.generateContent({
          ...options,
          model: currentModel,
        });
        return response;
      } catch (err: any) {
        lastError = err;
        const errStr = String(err?.message || err || "").toLowerCase();
        const is503OrRateLimit =
          errStr.includes("503") ||
          errStr.includes("unavailable") ||
          errStr.includes("high demand") ||
          errStr.includes("429") ||
          errStr.includes("rate limit") ||
          errStr.includes("quota") ||
          errStr.includes("overloaded") ||
          err?.status === 503 ||
          err?.status === 429;

        console.warn(`Attempt ${attempt} with model ${currentModel} failed:`, errStr);

        if (is503OrRateLimit && attempt < attempts) {
          // Wait briefly before retrying
          await new Promise((resolve) => setTimeout(resolve, attempt * 1200));
          continue;
        }

        // If it was a non-503 or we've run out of attempts, break to try the next model
        break;
      }
    }
  }

  throw lastError;
}

async function callChatWithFallback(
  ai: GoogleGenAI,
  options: {
    model: string;
    config?: any;
    history?: any[];
    message: string;
  }
) {
  const primaryModel = options.model;
  const backupModel = "gemini-3.1-flash-lite";

  const modelsToTry = [primaryModel, backupModel];
  let lastError: any = null;

  for (const currentModel of modelsToTry) {
    let attempts = 2;
    for (let attempt = 1; attempt <= attempts; attempt++) {
      try {
        console.log(`Executing Chat session with model: ${currentModel} (Attempt ${attempt}/${attempts})`);
        const chatSession = ai.chats.create({
          model: currentModel,
          config: options.config,
          history: options.history,
        });
        const response = await chatSession.sendMessage({ message: options.message });
        return response;
      } catch (err: any) {
        lastError = err;
        const errStr = String(err?.message || err || "").toLowerCase();
        const is503OrRateLimit =
          errStr.includes("503") ||
          errStr.includes("unavailable") ||
          errStr.includes("high demand") ||
          errStr.includes("429") ||
          errStr.includes("rate limit") ||
          errStr.includes("quota") ||
          errStr.includes("overloaded") ||
          err?.status === 503 ||
          err?.status === 429;

        console.warn(`Chat attempt ${attempt} with model ${currentModel} failed:`, errStr);

        if (is503OrRateLimit && attempt < attempts) {
          await new Promise((resolve) => setTimeout(resolve, attempt * 1200));
          continue;
        }
        break;
      }
    }
  }

  throw lastError;
}

// API to generate adaptive financial advice with Gemini
app.post("/api/analyze", async (req, res) => {
  try {
    const { 
      incomes, 
      expenses, 
      goals,
      allIncomes,
      allExpenses,
      activePeriod,
      connectPastMonths,
      connectFutureMonths,
      enableMultiPeriod
    } = req.body;

    // Validate request keys
    if (!incomes || !expenses || !goals) {
      return res.status(400).json({
        error: "Missing required properties: incomes, expenses, or goals are required.",
      });
    }

    let valAI;
    try {
      valAI = getAI();
    } catch (e: any) {
      // Handle missing API key gracefully by providing a standard, helpful response
      return res.status(200).json({
        recommendedRatio: { kebutuhan: 50, keinginan: 30, tabungan: 20 },
        summary: "Analisis simulasi aktif. Untuk menyalakan penasihat keuangan adaptif berbasis AI secara real-time, silakan tambahkan kunci API GEMINI_API_KEY Anda di Settings > Secrets.",
        suggestions: [
          "Gunakan rasio default 50/30/20 sebagai tolak ukur dasar keuangan Anda.",
          "Prioritaskan pengeluaran Kebutuhan (Needs) agar tidak melebih 50% pendapatan bersih.",
          "Tingkatkan alokasi tabungan untuk mempercepat realisasi target dana darurat Anda.",
          "Analisis pengeluaran Keinginan (Wants) secara berkala dan potong biaya langganan yang tidak terpakai."
        ],
        goalsAnalysis: goals.map((g: any) => ({
          goalId: g.id,
          feasibility: "Butuh Penyesuaian",
          advice: `Lakukan alokasi dana bulanan minimum secara konsisten untuk mencapai target ${g.targetAmount.toLocaleString("id-ID")} Anda.`
        })),
        urgencyLevel: "Waspada",
        isDemo: true
      });
    }

    const totalIncome = incomes.reduce((sum: number, inc: any) => sum + inc.amount, 0);
    const totalExpense = expenses.reduce((sum: number, exp: any) => sum + exp.amount, 0);

    const selectedMonth = activePeriod || "2026-05";
    const fullIncomes = allIncomes || incomes;
    const fullExpenses = allExpenses || expenses;

    // Handle user configuration of the multi-period chain 
    const isMultiActive = enableMultiPeriod !== false;
    const pastVal = typeof connectPastMonths === "number" ? connectPastMonths : 2;
    const futureVal = typeof connectFutureMonths === "number" ? connectFutureMonths : 2;

    const promptText = `
      Anda adalah Penasihat Keuangan Publik yang cerdas, empati, dan taktis dari Indonesia.
      Tugas Anda adalah melakukan analisis keuangan secara HOLISTIK dengan meninjau lini masa perkembangan finansial pengguna secara terhubung:
      - Periode Aktif saat ini yang sedang dievaluasi/dibuka: ${selectedMonth}
      ${isMultiActive ? `- STATUS RANTAI MULTI-PERIODE AKTIF: Tinjau dan hubungkan dengan data historis ${pastVal} bulan ke belakang (seperti maret/april dsb) serta ${futureVal} bulan rencana anggaran ke depan.` : `- STATUS RANTAI MULTI-PERIODE PENUH NONAKTIF: Fokus eksklusif hanya pada rincian saku bulan ${selectedMonth}.`}

      SITUASI PERIODE AKTIF TERPILIH SAAT INI (${selectedMonth}):
      1. Pendapatan Bulanan:
         Total: Rp ${totalIncome.toLocaleString("id-ID")}
         Detail: ${JSON.stringify(incomes)}
 
      2. Pengeluaran:
         Total: Rp ${totalExpense.toLocaleString("id-ID")}
         Detail: ${JSON.stringify(expenses)}

      ${isMultiActive ? `TREN PENUH SELURUH PERIODE (UNTUK MELIHAT PERKEMBANGAN):
      - Semua Pendapatan Lintas Waktu: ${JSON.stringify(fullIncomes)}
      - Semua Pengeluaran Lintas Waktu: ${JSON.stringify(fullExpenses)}` : ""}

      3. Tujuan Finansial (Goals):
         Detail: ${JSON.stringify(goals)}

      Mohon lakukan analisis mendalam terhadap seluruh data di atas secara terintegrasi:
      - Tentukan recommendedRatio (kebutuhan, keinginan, tabungan) dalam total persen (harus berjumlah tepat 100) untuk periode aktif (${selectedMonth}) secara ADAPTIF.
      - Berikan 'summary' ringkas berbahasa Indonesia yang ramah, sopan, tajam, dan edukatif. ${isMultiActive ? `Hubungkan tren lintas waktu yang terdeteksi dengan membandingkan parameter ${pastVal} bulan lalu dengan sisa target ${futureVal} bulan depan. Beritahu pengguna kelemahan pengeluaran impulsif atau apresiasi perbaikan sisa anggaran secara runut.` : `Fokuslah mengulas pola saku periode aktif ${selectedMonth} saja.`}
      - Sediakan minimal 3-5 saran tuntas ('suggestions') taktis yang didasari kondisi keuangan ini. ${isMultiActive ? `Beri rekomendasi bagaimana bersiap untuk mengamankan rencana saku bulan depan agar tidak goyah seperti riwayat bocor di bulan sebelumnya.` : `Saran harus terarah pada pembatasan pengeluaran kategori Keinginan periode ${selectedMonth}.`}
      - Evaluasi setiap tujuan finansial ('goalsAnalysis') dengan mencocokkan targetAmount dan targetDate dengan kapasitas sisa tabungan bulanan riil mereka dari tren keseluruhan. Nyatakan kelayakannya ('feasibility') dan beri masukan taktis spesifik.
      - Tentukan 'urgencyLevel' kesehatan keuangan mereka saat ini ('Aman', 'Waspada', atau 'Kritis') dengan menakar kestabilan tren terkini.
    `;

    const response = await callGenerateContentWithFallback(valAI, {
      model: "gemini-2.5-flash",
      contents: promptText,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            recommendedRatio: {
              type: Type.OBJECT,
              properties: {
                kebutuhan: { type: Type.INTEGER, description: "Rasio kebutuhan dalam persen" },
                keinginan: { type: Type.INTEGER, description: "Rasio keinginan dalam persen" },
                tabungan: { type: Type.INTEGER, description: "Rasio tabungan/investasi dalam persen" }
              },
              required: ["kebutuhan", "keinginan", "tabungan"]
            },
            summary: { type: Type.STRING, description: "Ringkasan analisis keuangan singkat penuh wawasan" },
            suggestions: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "Saran taktis atau aksi konkret untuk meningkatkan alokasi secara adaptif"
            },
            goalsAnalysis: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  goalId: { type: Type.STRING, description: "ID dari tujuan finansial" },
                  feasibility: { type: Type.STRING, description: "Mudah Tercapai | Sangat Menantang | Butuh Penyesuaian" },
                  advice: { type: Type.STRING, description: "Saran taktis spesifik untuk tujuan finansial tersebut" }
                },
                required: ["goalId", "feasibility", "advice"]
              }
            },
            urgencyLevel: { type: Type.STRING, description: "Status kesehatan: Aman | Waspada | Kritis" }
          },
          required: ["recommendedRatio", "summary", "suggestions", "goalsAnalysis", "urgencyLevel"]
        }
      }
    });

    const bodyText = response.text;
    if (!bodyText) {
      throw new Error("No response text received from Gemini server.");
    }

    const result = JSON.parse(bodyText.trim());
    res.json(result);
  } catch (error: any) {
    console.error("Gemini analysis error:", error);
    res.status(500).json({ error: error.message || "Something went wrong during analysis" });
  }
});

// API for interactive financial Q&A chat with saku/context grounding
app.post("/api/chat", async (req, res) => {
  try {
    const { 
      message, 
      history = [], 
      incomes = [], 
      expenses = [], 
      goals = [], 
      activePeriod = "2026-05" 
    } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Missing required 'message' field." });
    }

    let valAI;
    try {
      valAI = getAI();
    } catch (e: any) {
      // Return a friendly fallback guidance response if Gemini API key is not defined yet
      return res.json({
        text: `Halo! Saya adalah **AdaptSaku AI Assistant**. 

Saat ini saya berjalan dalam *Mode AdaptSaku Mandiri* karena kunci \`GEMINI_API_KEY\` belum terpasang di **Settings > Secrets**. 

Berdasarkan tinjauan anggaran aktif Anda:
- **Total Masuk**: Rp ${(incomes.reduce((s: number, i: any) => s + i.amount, 0)).toLocaleString("id-ID")}
- **Total Belanja**: Rp ${(expenses.reduce((s: number, e: any) => s + e.amount, 0)).toLocaleString("id-ID")}
- **Tujuan Impian**: ${goals.length} target terdaftar.

Ingin mengobrol interaktif secara cerdas dengan saya membahas strategi finansial Anda sesungguhnya? Cukup pasangkan kunci API di menu asisten pengelola!`,
        isDemo: true
      });
    }

    const totalIncome = incomes.reduce((s: number, i: any) => s + i.amount, 0);
    const totalExpense = expenses.reduce((s: number, e: any) => s + e.amount, 0);

    const systemInstruction = `
      Anda adalah "AdaptSaku AI Assistant", asisten keuangan personal berbasis AI yang cerdas, berempati tinggi, komunikatif, dan taktis dari Indonesia.
      Tugas utama Anda adalah membantu pengguna mengulas, merencanakan, serta mengelola keuangan/cashflow mereka secara bijak.
      
      Gunakan nada bicara yang sopan, ramah, interaktif (tidak kaku), edukatif, serta memotivasi pengguna agar konsisten menabung. Gunakan istilah keuangan lokal yang mudah dipahami kalangan umum di Indonesia.
      Anda menjawab pertanyaan berdasarkan konteks rekam keuangan riil mereka yang dilampirkan di bawah ini.

      SITUASI KEGIATAN AKTIF SAAT INI (Periode: ${activePeriod}):
      1. Pendapatan Bulan Ini:
         - Total Pemasukan: Rp ${totalIncome.toLocaleString("id-ID")}
         - Rincian Transaksi: ${JSON.stringify(incomes)}

      2. Pengeluaran Belanja Bulan Ini:
         - Total Pengeluaran: Rp ${totalExpense.toLocaleString("id-ID")}
         - Rincian Transaksi: ${JSON.stringify(expenses)}

      3. Sasaran Impian Finansial (Goals):
         - Rincian Sasaran: ${JSON.stringify(goals)}

      ATURAN KOMUNIKASI CHAT:
      - Selalu jawab pertanyaan dengan menyangkutpautkan (grounding) data di atas jika relevan dengan pertanyaan user (misalnya jumlah pemasukan, belanjaan, sisa anggaran tabungan).
      - Berikan saran konkret, misalnya rincian langkah menghemat biaya non-primer atau tips mengatur dana darurat.
      - Bila data di atas masih kosong atau Rp 0, sambut pengguna dengan ramah, jelaskan cara mudah menginput data transaksi pertama mereka di tab, dan tawarkan beberapa simulasi tips finansial mendasar.
      - Buat tanggapan Anda terstruktur menggunakan format Markdown (heading, list, teks tebal) agar nyaman dibaca di layar web app.
    `;

    // Process messaging history using Gemini's structured chat format
    const geminiHistory = history.map((item: any) => ({
      role: item.role === "user" ? "user" : "model",
      parts: [{ text: item.text }]
    }));

    const response = await callChatWithFallback(valAI, {
      model: "gemini-2.5-flash",
      config: {
        systemInstruction,
        temperature: 0.7,
      },
      history: geminiHistory,
      message,
    });
    const replyText = response.text;

    if (!replyText) {
      throw new Error("Asisten AI gagal merumuskan tanggapan.");
    }

    res.json({ text: replyText });

  } catch (error: any) {
    console.error("Gemini Chat API Error:", error);
    res.status(500).json({ error: error.message || "Terjadi kesalahan pada server asisten AI." });
  }
});

// API for processing transaction images using Gemini Multimodal
app.post("/api/analyze-receipt", async (req, res) => {
  try {
    const { imageBase64, mimeType, notes } = req.body;
    
    if (!imageBase64 || !mimeType) {
      return res.status(400).json({ error: "Missing imageBase64 or mimeType." });
    }

    let valAI;
    try {
      valAI = getAI();
    } catch (e: any) {
      return res.status(401).json({ error: "API key belum dikonfigurasi. Harap pasang GEMINI_API_KEY di Settings." });
    }

    const promptText = `
      Anda adalah "AdaptSaku AI Receipt Scanner". Tugas Anda adalah mengekstraksi data dari gambar bukti transaksi / kuitansi / struk / transfer.
      Gunakan informasi teks tambahan dari pengguna (jika ada): "${notes || 'Tidak ada catatan khusus'}".

      Ekstrak dan kembalikan struktur JSON data transaksi berikut:
      - title: Nama transaksi singkat, misalnya "Beli Kopi", "Bayar Kos", atau nama penerima/pengirim transfer. (string)
      - amount: Nominal angka (hanya angka positif bulat) yang tertera pada transaksi. (number)
      - expectedType: Klasifikasikan sebagai 'expense' (pengeluaran uang) atau 'income' (pemasukan uang). (string: 'expense' | 'income')
      - category:
         Jika 'expense', pilih antara: 'kebutuhan' | 'keinginan' | 'tabungan'
         Jika 'income', pilih antara: 'primary' | 'side-hustle' | 'investment' | 'other'
      - subcategory: (opsional) JIKA type adalah 'expense', pilih subkategori yang BENAR-BENAR COCOK dari daftar berikut sesuai category-nya:
         * kebutuhan: 'Makanan & Minuman', 'Sewa Tempat Tinggal / Kos', 'Listrik, Gas & Air', 'Transportasi / Bensin', 'Kesehatan / Obat', 'Cicilan Wajib/Hutang', 'Pendidikan', 'Pulsa & Internet'
         * keinginan: 'Belanja Mode / Skincare', 'Makan Mewah/Kafe (Nongkrong)', 'Streaming & Hiburan (Film, Games)', 'Liburan / Traveling', 'Hobi & Koleksi', 'Keanggotaan Gimnasium'
         * tabungan: 'Dana Darurat', 'Investasi Saham / Reksadana', 'Tabungan Emas / SBN', 'Investasi Crypto', 'Kas Berjangka / Deposito', 'Persiapan Menikah / Rumah'
         Jika Anda tidak yakin atau tidak ada yang pas, kembalikan 'Lainnya'. Untuk 'income', biarkan string kosong atau 'Lainnya'.
      - date: Format ISO string (contoh: "2026-05-27T10:30") atau kembalikan null jika tidak tertera jelas.

      Hanya kembalikan JSON Object murni yang valid sesuai skema.
    `;

    const response = await callGenerateContentWithFallback(valAI, {
      model: "gemini-2.5-pro",
      contents: [
        {
          inlineData: {
            data: imageBase64,
            mimeType: mimeType
          }
        },
        promptText
      ],
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            amount: { type: Type.INTEGER },
            expectedType: { type: Type.STRING },
            category: { type: Type.STRING },
            subcategory: { type: Type.STRING },
            date: { type: Type.STRING, nullable: true },
          },
          required: ["title", "amount", "expectedType", "category"]
        }
      }
    });

    const bodyText = response.text;
    if (!bodyText) {
      throw new Error("No response string from model.");
    }
    
    const result = JSON.parse(bodyText.trim());
    res.json(result);

  } catch (error: any) {
    console.error("Receipt Analysis Error:", error);
    res.status(500).json({ error: error.message || "Gagal memproses gambar bukti transaksi." });
  }
});

// Configure Vite middleware and SPA serving
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    // Production serving paths
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server fully running on http://localhost:${PORT}`);
  });
}

startServer();
