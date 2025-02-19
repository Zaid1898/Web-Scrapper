import dotenv from 'dotenv';
import puppeteer from 'puppeteer';
import { GoogleGenerativeAI } from "@google/generative-ai";
import { ChromaClient } from 'chromadb';

// Load environment variables
dotenv.config();

// Initialize services
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const chromaClient = new ChromaClient({ path: 'http://localhost:8000' });
const WEB_COLLECTION = 'WEB_SCRAPED_DATA_COLLECTION';

// Web scraping function (unchanged)
async function scrapeWebpage(url = '') {
    const browser = await puppeteer.launch({ 
        headless: true,
        args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36');

    await page.goto(url, { waitUntil: 'networkidle2', timeout: 30000 });

    let metaDescription = '';
    try {
        metaDescription = await page.$eval('meta[name="description"]', el => el.content);
    } catch (error) {
        console.log('Meta description not found');
    }

    const pageTitle = await page.title();
    const bodyText = await page.$eval('body', el => el.innerText.replace(/\s+/g, ' ').trim());

    await browser.close();

    return {
        title: pageTitle,
        metaDescription,
        body: bodyText,
    };
}

// Embedding generation using Gemini
async function generateVectorEmbeddings({ text }) {
    const model = genAI.getGenerativeModel({ model: "embedding-001" });
    const result = await model.embedContent(text);
    return result.embedding.values;
}

// ChromaDB integration (unchanged)
async function resetCollection() {
    try {
        await chromaClient.deleteCollection({ name: WEB_COLLECTION });
        console.log('🗑️ Collection reset');
    } catch (error) {
        // Ignore collection not found errors
    }
}

async function insertIntoDB({ embedding, url, metadata }) {
    const collection = await chromaClient.getOrCreateCollection({
        name: WEB_COLLECTION,
    });
    await collection.add({
        ids: [url],
        embeddings: [embedding],
        metadatas: [metadata],
    });
}

// Data ingestion
async function ingest(url = '') {
    const { title, metaDescription, body } = await scrapeWebpage(url);
    console.log('📖 Scraped Title:', title);
    
    const bodyEmbedding = await generateVectorEmbeddings({ text: body });
    
    await insertIntoDB({
        embedding: bodyEmbedding,
        url,
        metadata: { title, metaDescription, body },
    });

    console.log(`✅ Successfully ingested: ${url}`);
}

// Chat function
async function chat(question = '') {
    try {
        const questionEmbedding = await generateVectorEmbeddings({ text: question });
        
        const collection = await chromaClient.getOrCreateCollection({
            name: WEB_COLLECTION,
        });

        const queryResult = await collection.query({
            nResults: 1,
            queryEmbeddings: [questionEmbedding],
        });

        if (!queryResult.metadatas[0]?.length) {
            console.log("🤖: No relevant context found");
            return;
        }

        const { title, body } = queryResult.metadatas[0][0];
        const context = body.slice(0, 3000);

        const model = genAI.getGenerativeModel({
            model: "gemini-pro",
            safetySettings: [
                { category: "HARM_CATEGORY_HARASSMENT", threshold: "BLOCK_NONE" },
                { category: "HARM_CATEGORY_HATE_SPEECH", threshold: "BLOCK_NONE" }
            ]
        });

        const prompt = `Analyze this webpage content:
        Title: ${title}
        Content: ${context}
        
        Question: ${question}
        
        Answer using ONLY the content above. If unsure, say: "I couldn't find an answer."`;

        const result = await model.generateContent(prompt);
        const response = await result.response;
        
        console.log('🤖:', response.text());

    } catch (error) {
        console.error('Chat Error:', error);
    }
}

// Main function
async function main() {
    try {
        const url = 'http://books.toscrape.com/';
        await resetCollection();
        await ingest(url);
        await chat('Give me 550 words on this site?');
    } catch (error) {
        console.error('Main Error:', error.message);
    }
}

// Run
main();