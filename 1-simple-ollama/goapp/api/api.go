package api

import (
	"api_ollama/services/ask"
	"api_ollama/utils"
	"log"
	"net/http"

	"github.com/gorilla/mux"
)

type APIServer struct {
	addr string
}

func NewAPIServer(addr string) *APIServer {
	return &APIServer{addr: addr}
}

func (s *APIServer) Start() error {
	router := mux.NewRouter()
	subrouter := router.PathPrefix("/api").Subrouter()

	ask.RegisterRoutes(subrouter)

	router.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		utils.WriteJSON(
			w,
			http.StatusOK,
			map[string]string{"message": "App running"})
	}).Methods(http.MethodGet)

	log.Println("Listening on", s.addr)
	return http.ListenAndServe(s.addr, router)
}
