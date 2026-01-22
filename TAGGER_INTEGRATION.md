# Music Tagger Integration

The loop export step can optionally enrich exported loop metadata using a lightweight music tagger that:

- Reads basic tags from the source audio file (via mutagen)
- Optionally queries Spotify (client credentials) to improve title/artist matching

How it works:

- When `export_top_loops(...)` is called with `add_tags=True` (default), it calls `pipelines.music_tagger.identify_track(...)` once per source audio file and appends the following keys to each loop row in `loops_metadata.csv`:
  - `tag_title`
  - `tag_genres`
  - `tag_styles`
  - `tag_track_number`

## Configure Spotify (optional)

If you want better matching, create a `.env` file (or export env vars) with your Spotify client credentials:

```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
```

An example file is provided at `.env.example`.

If the credentials are not provided, the tagger will only use local file tags.

## Dependencies

New dependencies added for tagging:

- `mutagen`
- `rapidfuzz`
- `requests`
- `python-dotenv` (optional; handy for local `.env` usage)

These are already included in `requirements_dj_loop_pipeline.txt`.

## Using in code

If you are calling the exporter directly, the signature now includes an optional flag:

```
export_top_loops(df, audio_file, outdir, top_k=5, sr=44100, export_mp3=False, add_tags=True)
```

Set `add_tags=False` to skip tagging completely.
