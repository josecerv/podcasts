import os
import pandas as pd
import psycopg2
import json

# ----------------------------------------------------------------
# STEP 1: Load podcast IDs from the first and second survey CSV files.
# ----------------------------------------------------------------
# Read survey files; assumes there is a column named "podcastID"
df_email_one = pd.read_csv("email_one_survey.csv")
df_email_two = pd.read_csv("email_two_survey.csv")

# Extract the unique podcast IDs from both surveys and combine them
podcast_ids_one = df_email_one["podcastID"].dropna().unique().tolist()
podcast_ids_two = df_email_two["podcastID"].dropna().unique().tolist()
podcast_ids = list(set(podcast_ids_one + podcast_ids_two))  # Combine and remove duplicates
print(f"Found {len(podcast_ids)} unique podcast IDs in email one and two surveys.")

# ----------------------------------------------------------------
# STEP 2: Load publisher and description info from podcasts_final_sample.csv.
# ----------------------------------------------------------------
# Assumption: podcasts_final_sample.csv contains a column "podcastID" to join on.
df_podcasts = pd.read_csv("podcasts_final_sample.csv", dtype={"podcast_id": str, "publisher": str, "description": str})

# Set the index to "podcastID" to make lookup easier.
podcast_info = df_podcasts.set_index("podcast_id")[["publisher", "description"]].to_dict(orient="index")

# ----------------------------------------------------------------
# STEP 3: Connect to PostgreSQL and query for the three most recent episodes.
# ----------------------------------------------------------------
# Ensure PostgreSQL environment variables are set as provided.
os.environ["PGUSER"] = "jcervantez"
os.environ["PGPASSWORD"] = "Cervantez12"
os.environ["PGHOST"] = "localhost"
os.environ["PGPORT"] = "5432"
os.environ["PGDATABASE"] = "podcast_episodes"

# Connect to the PostgreSQL database using psycopg2.
try:
    conn = psycopg2.connect(
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        host=os.environ["PGHOST"],
        port=os.environ["PGPORT"],
        dbname=os.environ["PGDATABASE"]
    )
except Exception as e:
    print("Error connecting to the database:", e)
    exit(1)

cursor = conn.cursor()

# For this script, we assume that the table containing episodes is named "episodes".
# Adjust the table name if necessary.
episode_query = """
    SELECT episode_id, episode_title, episode_description 
    FROM episodes 
    WHERE podcast_id = %s 
    ORDER BY episode_published DESC 
    LIMIT 3;
"""

# Final collection that will store one JSON per podcast.
final_podcast_entries = []

for pid in podcast_ids:
    # Lookup publisher and description; default to None if not present.
    info = podcast_info.get(pid, {})
    publisher = info.get("publisher")
    description = info.get("description")
    
    # Retrieve the three most recent episodes for this podcast.
    try:
        cursor.execute(episode_query, (pid,))
        rows = cursor.fetchall()
    except Exception as e:
        print(f"Error querying episodes for podcast_id {pid}: {e}")
        rows = []
    
    # Construct episodes list as an array of dictionaries.
    episodes = []
    for row in rows:
        episode = {
            "episode_id": row[0],
            "title": row[1],  # Changed to match expected format in host extraction script
            "description": row[2]  # Changed to match expected format in host extraction script
        }
        episodes.append(episode)
    
    # Build the final JSON object for this podcast.
    podcast_entry = {
        "podcast_id": pid,
        "publisher": publisher,
        "description": description,
        "episodes": episodes
    }
    
    final_podcast_entries.append(podcast_entry)

# Close the database connections.
cursor.close()
conn.close()

# ----------------------------------------------------------------
# STEP 4: Write the data to a JSONL file.
# ----------------------------------------------------------------
output_filename = "email_one_two_host.jsonl"
with open(output_filename, "w", encoding="utf-8") as outfile:
    for entry in final_podcast_entries:
        json.dump(entry, outfile)
        outfile.write("\n")

print(f"JSONL file '{output_filename}' created successfully.")
