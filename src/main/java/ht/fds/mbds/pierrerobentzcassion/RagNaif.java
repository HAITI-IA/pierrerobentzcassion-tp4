package ht.fds.mbds.pierrerobentzcassion;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ht.fds.mbds.pierrerobentzcassion.llm.Assistant;

import java.nio.file.FileSystems;
import java.nio.file.PathMatcher;
import java.util.List;

public class RagNaif {
    public static void main(String[] args) {
        // Creation du chatModel
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .build();
        // 1. Creation du document Parser
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        // 2. créer un Document
        Document document = ClassPathDocumentLoader.loadDocument("rag.pdf", documentParser);

        // 3. Creation d'un document Splitter
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300,30,300);
        List < TextSegment> chunks = documentSplitter.split(document);

        EmbeddingStore embeddingStore = new InMemoryEmbeddingStore<>();
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(20))
                .contentRetriever(EmbeddingStoreContentRetriever.from(embeddingStore))
                .build();

    }
}
