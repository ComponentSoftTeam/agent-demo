import unittest
from unittest.mock import MagicMock, patch
from Compsoft_Gradio_Agent_app import exec_agent  # replace with the actual module name


class TestExecAgent(unittest.IsolatedAsyncioTestCase):
    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_empty_chatbot_default_prompt(
        self, mock_trace_list_append, mock_get_chain
    ):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = None
        session_id = "session_id"
        system_prompt = ""
        prompt = "I have no request"
        model_type = "mistral-large-latest"

        result = await exec_agent(
            chatbot, session_id, system_prompt, prompt, model_type
        )
        self.assertEqual(result, ([["I have no request", "response"]], ""))

    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_non_empty_chatbot_custom_prompt(
        self, mock_trace_list_append, mock_get_chain
    ):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = [["human", "ai"]]
        session_id = "session_id"
        system_prompt = "system prompt"
        prompt = "custom prompt"
        model_type = "mistral-large-latest"

        result = await exec_agent(
            chatbot, session_id, system_prompt, prompt, model_type
        )
        self.assertEqual(result, ([["human", "ai"], ["custom prompt", "response"]], ""))

    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_empty_system_prompt(self, mock_trace_list_append, mock_get_chain):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = None
        session_id = "session_id"
        system_prompt = ""
        prompt = "prompt"
        model_type = "mistral-large-latest"

        result = await exec_agent(
            chatbot, session_id, system_prompt, prompt, model_type
        )
        self.assertEqual(result, ([["prompt", "response"]], ""))

    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_non_empty_system_prompt(
        self, mock_trace_list_append, mock_get_chain
    ):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = None
        session_id = "session_id"
        system_prompt = "system prompt"
        prompt = "prompt"
        model_type = "mistral-large-latest"

        result = await exec_agent(
            chatbot, session_id, system_prompt, prompt, model_type
        )
        self.assertEqual(result, ([["prompt", "response"]], ""))

    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_default_model_type(self, mock_trace_list_append, mock_get_chain):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = None
        session_id = "session_id"
        system_prompt = ""
        prompt = "prompt"
        model_type = "mistral-large-latest"

        result = await exec_agent(
            chatbot, session_id, system_prompt, prompt, model_type
        )
        self.assertEqual(result, ([["prompt", "response"]], ""))

    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_custom_model_type(self, mock_trace_list_append, mock_get_chain):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = None
        session_id = "session_id"
        system_prompt = ""
        prompt = "prompt"
        model_type = "custom-model-type"

        result = await exec_agent(
            chatbot, session_id, system_prompt, prompt, model_type
        )
        self.assertEqual(result, ([["prompt", "response"]], ""))

    @patch("Compsoft_Gradio_Agent_app.get_chain")
    @patch("Compsoft_Gradio_Agent_app.trace_list_append")
    async def test_invalid_session_id(self, mock_trace_list_append, mock_get_chain):
        mock_agent_chain = MagicMock()
        mock_get_chain.return_value = mock_agent_chain
        mock_agent_chain.ainvoke.return_value = "response"

        chatbot = None
        session_id = None
        system_prompt = ""
        prompt = "prompt"
        model_type = "mistral-large-latest"

        with self.assertRaises(TypeError):
            await exec_agent(chatbot, session_id, system_prompt, prompt, model_type)


if __name__ == "__main__":
    unittest.main()
