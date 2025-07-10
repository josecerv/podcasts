// Use the v2 version of firebase-functions
const functions = require("firebase-functions/v2");
const logger = require("firebase-functions/logger");
const admin = require("firebase-admin");

admin.initializeApp();

// Initialize Firestore
const db = admin.firestore();

// Cloud Function to get podcast data (Gen 2)
exports.getPodcastData = functions
    .https.onRequest(async (req, res) => {
    // Set CORS headers
      res.set("Access-Control-Allow-Origin", "*");
      res.set("Access-Control-Allow-Methods", "GET");

      // Handle preflight requests
      if (req.method === "OPTIONS") {
        res.set("Access-Control-Allow-Headers", "Content-Type");
        return res.status(204).send("");
      }

      try {
        const podcastId = req.query.podcastID;
        if (!podcastId) {
          return res.status(400).json({error: "podcastID parameter required"});
        }

        const doc = await db.collection("podcasts").doc(podcastId).get();
        if (doc.exists) {
          const podcast = doc.data();

          const episodesArray = (podcast.episodes || []).map((ep) => ({
            episode_id: ep.episode_id || "",
            title: (ep.title || "").trim(),
            date: ep.published_at || "",
            description: (ep.description || "")
                .replace(/<[^>]*>/g, "")
                .replace(/[\n\r]+/g, " ")
                .replace(/\s+/g, " ")
                .trim(),
          }));

          const responseData = {
            podcast_id: podcast.podcast_id || "",
            title: podcast.title || "",
            email: podcast.email || "",
            publishing_interval: podcast.publishing_interval || "",
            episodes_json: JSON.stringify(episodesArray),
            episode_count: episodesArray.length,
          };

          return res.status(200).json(responseData);
        } else {
          return res.status(404).json({error: "Podcast not found"});
        }
      } catch (error) {
        logger.error("Error fetching podcast data:", error);
        return res.status(500).json({error: "Internal Server Error"});
      }
    });
