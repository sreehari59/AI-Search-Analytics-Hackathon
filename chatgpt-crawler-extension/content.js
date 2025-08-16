console.log("ChatGPT Crawler Extension - Content Script Loaded");

document.addEventListener('keydown', (event) => {
    const activeElement = document.activeElement;

    if (event.key === 'Enter' && activeElement && activeElement.id === 'prompt-textarea') {
        console.log("Enter Clicked");

        setTimeout(async () => {
            const turns = document.querySelectorAll('[data-testid^="conversation-turn-"]');
            if (turns.length < 2) return;

            const latestTurnUsr = turns[turns.length - 2];
            const latestTurnAss = turns[turns.length - 1];

            const userPromptElement = latestTurnUsr.querySelector('div[data-message-author-role="user"]');
            const assistantWrapper = latestTurnAss.querySelector('div[data-message-author-role="assistant"]');

            if (!userPromptElement || !assistantWrapper) return;

            const userPrompt = userPromptElement.textContent.trim();
            const machineId = localStorage.getItem('machineId') || 'unknown_machine';
            const chatGroupId = new URLSearchParams(window.location.search).get('chat_id') || 'unknown_chat';

            console.log("✅ User Prompt:", userPrompt);

            // 🔹 Send prompt to server
            await fetch("https://tradelogsai.eastus.cloudapp.azure.com/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    machineId,
                    contentType: "prompt",
                    content: userPrompt,
                    chatGroupId,
                    sources: []
                })
            }).then(res => {
                if (res.ok) {
                    console.log("📤 Prompt sent successfully");
                } else {
                    console.warn("⚠️ Failed to send prompt");
                }
            });

            console.log("⏳ Waiting for assistant's response...");

            const responseElement = assistantWrapper.querySelector('.markdown');
            if (responseElement && responseElement.textContent.length > 0) {
                const gptResponse = responseElement.textContent.trim();
                console.log("✅ Assistant Response:", gptResponse);

                // 🔹 Send response to server
                try {
                    const res = await fetch("https://tradelogsai.eastus.cloudapp.azure.com/process", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            machineId,
                            contentType: "response",
                            content: gptResponse,
                            chatGroupId,
                            sources: []
                        })
                    });

                    if (res.ok) {
                        const data = await res.json();
                        console.log("Data",data)
                        if (Array.isArray(data)) {
                            console.log("HERE")
                            injectProductLinks(latestTurnAss, data);
                        } else {
                            console.warn("⚠️ Response JSON not an array");
                        }
                    } else {
                        console.warn("⚠️ Failed to send response");
                    }
                } catch (error) {
                    console.error("❌ Error sending GPT response:", error);
                }
            }

        }, 5000); // slight delay to allow DOM update
    }
});


function injectProductLinks(conversationTurnElement, products) {
    if (!products || products.length === 0) {
        console.log("No products to inject.");
        return;
    }

    const container = document.createElement("div");
    container.style.cssText = `
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid #e0e0e0;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
    `;

    const heading = document.createElement("h3");
    heading.textContent = "Recommended Links:";
    heading.style.cssText = `
        color: #e0e0e0;
        font-size: 1.1em;
        margin-bottom: 10px;
        grid-column: 1 / -1; /* Span across all columns */
    `;
    container.appendChild(heading);

    products.forEach(product => {
        const card = document.createElement("div");
        card.style.cssText = `
            background-color: #f9f9f9;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        `;

        card.onmouseover = () => {
            card.style.transform = 'translateY(-3px)';
            card.style.boxShadow = '0 4px 10px rgba(0,0,0,0.1)';
        };

        card.onmouseout = () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = '0 2px 5px rgba(0,0,0,0.05)';
        };

        const title = document.createElement("h4");
        // Attempt to extract a more user-friendly title
        let productTitle = "Recommended Product";
        try {
            const url = new URL(product.productUrl);
            const targetUrlParam = url.searchParams.get('target_url');
            if (targetUrlParam) {
                const decodedUrl = decodeURIComponent(targetUrlParam);
                // Extracting a "name" from the URL path, e.g., /products/awesome-item
                const pathParts = decodedUrl.split('/');
                const lastPart = pathParts[pathParts.length - 1];
                if (lastPart) {
                    productTitle = lastPart.replace(/-/g, ' ').replace(/\.html|\.php/i, '').trim();
                    if (productTitle.length > 30) { // Keep titles concise
                        productTitle = productTitle.substring(0, 30) + "...";
                    }
                    productTitle = productTitle.split('?')[0]; // Remove query params from title
                }
            }
        } catch (e) {
            console.warn("Could not parse product URL for title:", e);
        }
        title.textContent = productTitle.split('?')[0] || "Recommended Link";
        title.style.cssText = `
            font-size: 1em;
            margin-bottom: 8px;
            color: #0056b3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        `;
        card.appendChild(title);

        const description = document.createElement("p");
        description.textContent = "Click to learn more about this recommendation.";
        description.style.cssText = `
            font-size: 0.85em;
            color: #666;
            margin-bottom: 12px;
            flex-grow: 1; /* Allows description to take up available space */
        `;
        card.appendChild(description);

        const link = document.createElement("a");
        link.href = product.productUrl;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = "View Link";
        link.style.cssText = `
            display: inline-block;
            padding: 8px 15px;
            background-color: #007bff;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9em;
            align-self: flex-start; /* Aligns button to the start of its flex container */
            transition: background-color 0.2s ease-in-out;
        `;
        link.onmouseover = () => link.style.backgroundColor = '#0056b3';
        link.onmouseout = () => link.style.backgroundColor = '#007bff';

        card.appendChild(link);
        container.appendChild(card);
    });

    const groupElement = conversationTurnElement.querySelector('.group\\/conversation-turn');
    if (groupElement) {
        groupElement.appendChild(container);
        console.log("🧩 Product links injected:", products);
    } else {
        console.warn("❌ Could not find .group/conversation-turn element to inject product links.");
    }
}