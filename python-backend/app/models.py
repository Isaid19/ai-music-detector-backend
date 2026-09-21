"""Pydantic models for request/response validation."""

from typing import Optional

from pydantic import BaseModel, Field


class AnalysisResponse(BaseModel):
    audio_file: str
    scale: dict
    ai_score: float
    ai_score_bar: str
    classification: str
    decision_layer: str
    decision_reasoning: str
    shazam_analysis: dict
    metadata_analysis: dict
    acoustic_and_lyrics_analysis: dict
    youtube_analysis: dict
    Zero_Crossing_Rate: dict = Field(alias="Zero Crossing Rate")
    Spectral_Centroid: dict = Field(alias="Spectral Centroid")
    Spectral_Bandwidth: dict = Field(alias="Spectral Bandwidth")
    Spectral_Contrast: dict = Field(alias="Spectral Contrast")
    Spectral_Rolloff: dict = Field(alias="Spectral Rolloff")
    Spectral_Flux: dict = Field(alias="Spectral Flux")
    MFCCs: dict
    RMSE: dict

    class Config:
        populate_by_name = True


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str = "1.0.0"
