"use client"
import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import { motion, AnimatePresence } from 'framer-motion'

export default function Home() {
  const [question, setQuestion] = useState('')
  const [responses, setResponses] = useState([])
  const [socket, setSocket] = useState(null)
 // @ts-ignore
  useEffect(() => {
    const newSocket = io('http://localhost:3001')
     // @ts-ignore
    setSocket(newSocket)

    newSocket.on('connect', () => {
      console.log('Connected to server')
    })

    newSocket.on('response', (response) => {
       // @ts-ignore
      setResponses(prev => [...prev, response])
    })

    newSocket.on('error', (error) => {
      console.error('Error:', error)
    })

    return () => newSocket.disconnect()
  }, [])

  const sendQuestion = () => {
    if (socket && question.trim()) {
      // @ts-ignore
      socket.emit('query', question)
      setQuestion('')
    }
  }

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden p-6">
        <div className="flex gap-2 mb-6">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Zadaj pytanie..."
            onKeyPress={(e) => e.key === 'Enter' && sendQuestion()}
            className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
          />
          <motion.button
            onClick={sendQuestion}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
          >
            Wyślij
          </motion.button>
        </div>
        
        <div className="space-y-2">
          <AnimatePresence>
            {responses.map((response, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="p-3 bg-blue-50 rounded-lg text-gray-800"
              >
                {response}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}