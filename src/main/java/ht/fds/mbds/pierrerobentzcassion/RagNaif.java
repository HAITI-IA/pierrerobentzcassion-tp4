package ht.fds.mbds.pierrerobentzcassion;


import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.ClassPathDocumentLoader;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ht.fds.mbds.pierrerobentzcassion.llm.Assistant;

import java.nio.file.FileSystems;
import java.nio.file.PathMatcher;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

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
        DocumentSplitter documentSplitter = DocumentSplitters.recursive(300,30);
        List < TextSegment> segments = documentSplitter.split(document);
        // 4. Création d'un modèle d'embedding.
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        // 5. Créer les embeddings pour les segments.
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        // 6. Ajouter les embeddings et les segments associés dans un magasin d'embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        //Phase 2 du RAG
        // 1. Création du ContentRetriever.
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2) // Je precise que je veux deux resultats.
                .minScore(0.5) // score supérieur à 0.5 ou score minimal 0.5 pour similarité
                .build();

        // 2. Créez une mémoire pour 10 messages.
        MessageWindowChatMemory chatMemory = MessageWindowChatMemory.builder()
                .maxMessages(10)
                .build();
        // 3.Création de l'assistant avec le pattern builder.
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(chatMemory)
                .contentRetriever(contentRetriever)
                .build();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                System.out.println("==================================================");
                if ("fin".equalsIgnoreCase(question)) {
                    break;
                }
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
                System.out.println("==================================================");
            }
        }


    }
}
