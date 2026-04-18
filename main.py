import os
import pickle
from typing import Any, Dict, List, Optional,Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  
from dotenv import load_dotenv


load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_IMG_500= "https://image.tmdb.org/t/p/w500"
TMDB_BASE= "https://api.themoviedb.org/3"

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY not found in environment variables")
app= FastAPI(title="Movie Recommendation API", description="API for movie recommendations based on user preferences", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


DF_PATH = os.path.join(BASE_DIR, "data", "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "data", "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "data", "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "data", "tfidf.pkl")


df:Optional[pd.DataFrame] = None
indices_obj:Any = None
tfidf_matrix:Any = None
tfidf_obj:Any = None    

TITLE_TO_IDX:Optional[Dict[str, int]] = None

class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict]=[]


class TMDBMovieDetail(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_path: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict]=[]
    runtime: Optional[int] = None
    rating: Optional[float] = None

class TFIDFRecItem(BaseModel):
    title: str
    score: float
     
class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetail
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]



def _norm_title(title:str)->str:
    return str(title).strip().lower()

def make_img_url(path:Optional[str])->Optional[str]:
    if path:
        return f"{TMDB_IMG_500}{path}"
    return None  

async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{TMDB_BASE}{path}", params=q)
            response.raise_for_status()
            return response.json()

    # ✅ Network error (no internet, DNS fail, etc.)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to TMDB (check internet)"
        )

    # ✅ Other request errors
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Request error: {str(e)}"
        )

    # ✅ API returned bad status (like 401, 404)
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"TMDB API error: {e.response.text}"
        )

    # ✅ Catch-all
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

async def tmdb_cards_from_results(results:List[dict],limit:int=20)->List[TMDBMovieCard]:


    out:List[TMDBMovieCard] = []
    for item in (results or [])[:limit]:
        out.append(TMDBMovieCard(
            tmdb_id=item.get("id"),
            title=item.get("title"),
            overview=item.get("overview"),
            release_date=item.get("release_date"),
            poster_path= make_img_url(item.get("poster_path")),
            backdrop_url= make_img_url(item.get("backdrop_path")),
           genres=[{"id": gid} for gid in item.get("genre_ids", [])]
        )
        )
    return out



async def tmdb_movie_detail(tmdb_id:int)->TMDBMovieDetail:
    data = await tmdb_get(f"/movie/{tmdb_id}", {})
    return TMDBMovieDetail(
        tmdb_id=data.get("id"),
        title=data.get("title"),
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_path= make_img_url(data.get("poster_path")),
        backdrop_url= make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []),
        runtime=data.get("runtime"),
        rating=data.get("vote_average")
    )



async def tmdb_search_movies(query:str,page:int=1)->List[TMDBMovieCard]:    
    data = await tmdb_get("/search/movie", {"query": query, "page": page, "language": "en-US"})
    return await tmdb_cards_from_results(data.get("results", []), limit=10) 
  


async def tmdb_movies_by_genre(genre_id:int)->List[TMDBMovieCard]:
    data = await tmdb_get("/discover/movie", {"with_genres": genre_id, "sort_by": "popularity.desc"})
    return await tmdb_cards_from_results(data.get("results", []), limit=10) 


async def tmdb_movies_search_first(query: str) -> Optional[TMDBMovieCard]:
    results = await tmdb_search_movies(query)
    return results[0] if results else None


def build_title_index(indices:Any)->Dict[str, int]:
    title_to_idx:Dict[str, int] = {}
    if isinstance(indices,dict):
        for k,v in indices.items():
            title_to_idx[_norm_title(k)] = v
        return title_to_idx
    
    try:
        for k,v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception as e:
        raise RuntimeError(f"Error building title index: {str(e)}")
      





def get_local_idx_by_title(title:str)->Optional[int]:
    if not TITLE_TO_IDX:
        raise RuntimeError("Title index not loaded")
    return TITLE_TO_IDX.get(_norm_title(title))



def tfidf_recommended_titles(query_title:str, top_n:int=10)->List[Tuple[str, float]]:
    global df,tfidf_matrix
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="Recommendation data not loaded")
    idx=get_local_idx_by_title(query_title)
    if idx is None:
        return []
    #query vector
    qv=tfidf_matrix[idx]
    scores=(tfidf_matrix @ qv.T).toarray().ravel()

    # sort  descending 
    order =np.argsort(-scores)
    out:List[Tuple[str,float]]=[]
    for i in order:
        if int(i)==int(idx):
            continue
        try:
            title_i=str(df.iloc[int(i)]["title"])
        except Exception as e:
            continue
        out.append((title_i,float(scores[int(i)])))
        if len(out)>=top_n:
            break
        
    return out

async def attach_tmdb_card_by_title(title:str)->Optional[TMDBMovieCard]:
    try:
        m=await tmdb_movies_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=m.get("id"),
            title=m.get("title"),
            overview=m.get("overview"),
            release_date=m.get("release_date"),
            poster_path= make_img_url(m.get("poster_path")),
            backdrop_url= make_img_url(m.get("backdrop_path")),
            genres=[{"id": gid} for gid in m.get("genre_ids", [])]
        )
    except Exception:
        return None



#startup :load pickles
@app.on_event("startup")
def load_pickles():
    global df,indices_obj,tfidf_matrix,tfidf_obj,TITLE_TO_IDX
    with open(DF_PATH, "rb") as f:
        df = pickle.load(f)
    with open(INDICES_PATH, "rb") as f:
        indices_obj = pickle.load(f)
    with open(TFIDF_MATRIX_PATH, "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf_obj = pickle.load(f)
    TITLE_TO_IDX = build_title_index(indices_obj)
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a 'title' column")                  

#routes
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/home", response_model=List[TMDBMovieCard])
async def home(  
    category:str=Query("popular", description="Category of movies to fetch: popular, top_rated, upcoming"),
    limit:int=Query(24,ge=1,le=50)):
    
    try:
        if category == "trending":
            data= await tmdb_get("/trending/movie/day",{"language":"en-US" })
            return await tmdb_cards_from_results(data.get("results",[]),limit=limit)
        if category not in {"popular","top_rated","upcoming"}:
            raise HTTPException(status_code=400, detail="Invalid category")
        data= await tmdb_get(f"/movie/{category}",{"language":"en-US", "page":1})
        return await tmdb_cards_from_results(data.get("results",[]),limit=limit)
    except httpx.RequestError :
        raise 
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error fetching movies: {str(e)}") 



@app.get("/tmdb/search")
async def tmdb_search(
    query:str=Query(...,min_length=1),
    page:int =Query(1, ge=1),):
    return await tmdb_search_movies(query=query,page=page)

 

@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetail)
async def movie_detail(tmdb_id:int):
    try:
        return await tmdb_movie_detail(tmdb_id=tmdb_id)
    except httpx.RequestError as e :
        raise HTTPException(status_code=503,detail=f"Error coonecting to TMDB API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error fetching movie details: {str(e)}")
    

@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    try:
        details = await tmdb_movie_detail(tmdb_id)

        if not details.genres:
            return []

        # Use multiple genres
        genre_ids = [g["id"] for g in details.genres]

        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": ",".join(map(str, genre_ids)),
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )

        # Fetch extra to avoid losing count
        cards = await tmdb_cards_from_results(
            discover.get("results", []),
            limit=limit + 5
        )

        filtered = [c for c in cards if c.tmdb_id != tmdb_id]

        return filtered[:limit]

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"TMDB API error: {str(e)}"
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )

@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title:str=Query(...,min_length=1),
    top_n:int=Query(10,ge=1,le=50),
):
    resc=tfidf_recommended_titles(title,top_n=top_n)
    return [{"title":t,"score":s}for t,s in resc]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    best = await tmdb_movies_search_first(query)

    if not best:
        raise HTTPException(
            status_code=404,
            detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = best.tmdb_id
    details = await tmdb_movie_detail(tmdb_id)

    # TF-IDF
    try:
        tfidf_recs = tfidf_recommended_titles(
            details.title,
            top_n=tfidf_top_n
        )
    except Exception:
        tfidf_recs = []

    # Genre
    genre_recs = (
        await tmdb_movies_by_genre(details.genres[0]["id"])
        if details.genres else []
    )

    genre_recs = [
        r for r in genre_recs if r.tmdb_id != tmdb_id
    ][:genre_limit]

    return {
    "query": query,
    "movie_details": details,
    "tfidf_recommendations": [{"title": t, "score": s} for t, s in tfidf_recs],
    "genre_recommendations": genre_recs,
}

