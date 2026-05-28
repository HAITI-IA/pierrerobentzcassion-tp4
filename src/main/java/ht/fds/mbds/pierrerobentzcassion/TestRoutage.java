package ht.fds.mbds.pierrerobentzcassion;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.HashMap;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {
    private static ContentRetriever getContentRetriever(String fileName, EmbeddingStore<TextSegment> embeddingStore) {
        // 1. Creation du document Parser
        DocumentParser documentParser = new ApacheTikaDocumentParser();
        // 2. créer un Document
        Document document = ClassPathDocumentLoader.loadDocument(fileName, documentParser);

        // 3. Creation d'un document Splitter
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300,30);
        List< TextSegment> segments = documentSplitter.split(document);
        // 4. Création d'un modèle d'embedding.
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        // 5. Créer les embeddings pour les segments.
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        // 6. Ajouter les embeddings et les segments associés dans un magasin d'embeddings en mémoire
        embeddingStore.addAll(embeddings, segments);
        //Phase 2 du RAG
        // 1. Création du ContentRetriever.
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // Je precise que je veux deux resultats.
                .minScore(0.5) // score supérieur à 0.5 ou score minimal 0.5 pour similarité
                .build();

    }
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }
    public static void main(String[] args) {
        configureLogger();
        // Creation du chatModel
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(System.getenv("GEMINI_KEY"))
                .modelName("gemini-2.5-flash")
                .logRequestsAndResponses(true)
                .build();
        // phase 1: Creation des deux embeddingstore
        EmbeddingStore<TextSegment> store1 = new InMemoryEmbeddingStore<>();
        EmbeddingStore<TextSegment> store2 = new InMemoryEmbeddingStore<>();

        // phase 2 : Recuperation de textes les plus pertinents
        ContentRetriever retriever1 = getContentRetriever("rag.pdf", store1);
        ContentRetriever retriever2 = getContentRetriever("POWER - Les 48 lois de pouvoir - Robert Greene.pdf", store2);


        // Creation du querryRouter
        //Creation descriptions
        HashMap<ContentRetriever, String> descriptions = new HashMap<>();
        descriptions.put(retriever1, "Ce retriever contient des informations sur le RAG");
        descriptions.put(retriever2, "Ce retriever contient des informations sur le pouvoir");
        LanguageModelQueryRouter queryRouter = new LanguageModelQueryRouter(model, descriptions);

         // Creation d'un retrievalAugmentor
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();



    }
}
