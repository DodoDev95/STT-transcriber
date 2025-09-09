package main

import (
	"encoding/binary"
	"log"
	"math"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

const (
	sampleRate   = 16000
	frameSamples = 320
	frameBytes   = frameSamples * 2
	sttURL       = "ws://localhost:9000/ws" // Python STT
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true }, // tighten in prod
}

type meterMsg struct {
	Type       string  `json:"type"`
	Seq        uint64  `json:"seq"`
	Bytes      int     `json:"bytes"`
	TotalBytes uint64  `json:"total_bytes"`
	Seconds    float64 `json:"seconds"`
	RMS        float64 `json:"rms"`
	Peak       float64 `json:"peak"`
	RMSdBFS    float64 `json:"rms_dbfs"`
	PeakdBFS   float64 `json:"peak_dbfs"`
	Timestamp  int64   `json:"ts_ms"`
}

func main() {
	http.HandleFunc("/ws", wsHandler)
	log.Println("Gateway on :8080/ws  (proxy to", sttURL, ")")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func wsHandler(w http.ResponseWriter, r *http.Request) {
	// 1) Upgrade browser connection
	brConn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, "upgrade failed", http.StatusInternalServerError)
		return
	}
	defer brConn.Close()

	// 2) Dial upstream Python STT
	u, _ := url.Parse(sttURL)

	reqHeader := http.Header{}
	reqHeader.Set("Origin", "http://localhost:3000")

	upConn, _, err := websocket.DefaultDialer.Dial(u.String(), reqHeader)
	if err != nil {
		log.Println("dial STT failed:", err)
		_ = brConn.WriteJSON(map[string]any{"type": "error", "message": "stt unavailable"})
		return
	}
	defer upConn.Close()

	log.Println("client connected; proxying to STT")

	var seq uint64
	var totalBytes uint64
	var writeMu sync.Mutex // serialize writes to browser

	// 3) Goroutine: read FROM STT (text JSON) → forward TO browser
	go func() {
		for {
			mt, data, err := upConn.ReadMessage()
			if err != nil {
				log.Println("[GW] STT read error:", err)
				writeMu.Lock()
				_ = brConn.WriteJSON(map[string]any{"type": "info", "message": "stt closed"})
				writeMu.Unlock()
				_ = brConn.Close()
				return
			}

			// Log what we got from STT
			switch mt {
			case websocket.TextMessage:
				// JSON from Python (info/partial/final)
				preview := data
				if len(preview) > 200 {
					preview = preview[:200]
				}
				log.Printf("[GW] from STT: TEXT %dB %s", len(data), string(preview))
			case websocket.BinaryMessage:
				log.Printf("[GW] from STT: BINARY %dB", len(data))
			default:
				log.Printf("[GW] from STT: mt=%d %dB", mt, len(data))
			}

			// Forward to browser
			writeMu.Lock()
			if err := brConn.WriteMessage(mt, data); err != nil {
				writeMu.Unlock()
				log.Println("[GW] write to browser failed:", err)
				return
			}
			writeMu.Unlock()
		}
	}()

	// 4) Loop: read FROM browser → (a) meter to browser, (b) forward binary to STT
	for {
		mt, data, err := brConn.ReadMessage()
		if err != nil {
			// browser closed; tell STT
			_ = upConn.WriteControl(websocket.CloseMessage, []byte{}, time.Now().Add(time.Second))
			return
		}

		switch mt {
		case websocket.BinaryMessage:
			log.Printf("[GW] from browser: %d bytes", len(data))
			if err := upConn.WriteMessage(websocket.BinaryMessage, data); err != nil {
				log.Println("[GW] upstream write error:", err)
				return
			}

			// compute meter locally and send to browser
			tb := atomic.AddUint64(&totalBytes, uint64(len(data)))
			seconds := (float64(tb) / 2.0) / sampleRate
			rms, peak := meterPCM16LE(data)

			m := meterMsg{
				Type:       "meter",
				Seq:        atomic.AddUint64(&seq, 1),
				Bytes:      len(data),
				TotalBytes: tb,
				Seconds:    seconds,
				RMS:        rms, Peak: peak,
				RMSdBFS: toDBFS(rms), PeakdBFS: toDBFS(peak),
				Timestamp: time.Now().UnixMilli(),
			}
			writeMu.Lock()
			_ = brConn.WriteJSON(m)
			writeMu.Unlock()

		case websocket.TextMessage:
			// optional: forward text commands to STT (not needed now)
			_ = upConn.WriteMessage(websocket.TextMessage, data)
		}
	}
}

func toDBFS(x float64) float64 {
	if x <= 0 {
		return -120
	}
	return 20 * math.Log10(x)
}

func meterPCM16LE(b []byte) (rms, peak float64) {
	if len(b) < 2 {
		return 0, 0
	}
	var sumSq float64
	var maxAbs float64
	for i := 0; i+1 < len(b); i += 2 {
		s := int16(binary.LittleEndian.Uint16(b[i : i+2]))
		f := float64(s) / 32768.0
		af := math.Abs(f)
		if af > maxAbs {
			maxAbs = af
		}
		sumSq += f * f
	}
	n := float64(len(b) / 2)
	if n > 0 {
		rms = math.Sqrt(sumSq / n)
	}
	peak = maxAbs
	return
}
