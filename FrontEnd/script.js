const API_URL = "http://13.60.85.233:8000/recommend";

const queryInput = document.getElementById("queryInput");
const topK = document.getElementById("topK");
const topKValue = document.getElementById("topKValue");
const findBtn = document.getElementById("findBtn");
const resultsDiv = document.getElementById("results");
const statusDiv = document.getElementById("status");

const franchiseOnly = document.getElementById("franchiseOnly");
const useGenres = document.getElementById("useGenres");
const safeMode = document.getElementById("safeMode");
const usePopularity = document.getElementById("usePopularity");

topK.addEventListener("input", () => {
  topKValue.textContent = topK.value;
});

function inferSearchMode(query) {
  const trimmed = query.trim();
  if (!trimmed) return "Movie title";
  return trimmed.split(/\s+/).length <= 2
    ? "Movie title"
    : "Describe a movie vibe / story";
}

function getSelectedSearchMode(query) {
  const selected = document.querySelector('input[name="searchMode"]:checked');
  return selected ? selected.value : inferSearchMode(query);
}

function renderResults(results) {
  if (!results || results.length === 0) {
    resultsDiv.innerHTML = `<p class="info">No results found.</p>`;
    return;
  }

  resultsDiv.className = "results-grid";

  resultsDiv.innerHTML = results
    .map((movie) => {
      const poster = movie.poster_url || "https://via.placeholder.com/220x330?text=No+Image";
      const title = movie.title || "Untitled";
      const genres = movie.genres ? `Genres: ${movie.genres}` : "";
      const franchise = movie.franchise ? ` • Franchise: ${movie.franchise}` : "";
      const overview = movie.overview || "";

      return `
        <div class="movie-card">
          <img class="movie-poster" src="${poster}" alt="${title}" />
          <div>
            <div class="movie-title">${title}</div>
            <div class="movie-meta">${genres}${franchise}</div>
            <div class="movie-overview">${overview}</div>
          </div>
        </div>
      `;
    })
    .join("");
}

findBtn.addEventListener("click", async () => {
  const query = queryInput.value.trim();

  if (!query) {
    statusDiv.innerHTML = `<p class="error">Please enter a movie title or describe what you want to watch.</p>`;
    resultsDiv.innerHTML = "";
    return;
  }

  const payload = {
    query,
    franchise_only: franchiseOnly.checked,
    safe_mode: safeMode.checked,
    use_genres: useGenres.checked,
    use_popularity: usePopularity.checked,
    top_k: Number(topK.value),
    search_mode: getSelectedSearchMode(query),
  };

  statusDiv.innerHTML = `<p class="info">Thinking... finding the best matches for you.</p>`;
  resultsDiv.innerHTML = "";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text();
      statusDiv.innerHTML = `<p class="error">API error ${response.status}: ${text}</p>`;
      return;
    }

    const data = await response.json();
    statusDiv.innerHTML = `<p class="info">Found ${data.results?.length || 0} recommendations.</p>`;
    renderResults(data.results || []);
  } catch (error) {
    statusDiv.innerHTML = `<p class="error">Could not connect to API: ${error}</p>`;
  }
});